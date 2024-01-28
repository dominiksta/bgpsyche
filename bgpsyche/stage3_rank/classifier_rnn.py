import logging
from statistics import mean
from types import FrameType
import typing as t
import signal
import random

import torch
from torch import nn
from torcheval.metrics.functional import binary_accuracy, binary_f1_score
from bgpsyche.stage2_enrich.enrich import enrich_asn, enrich_path
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.stage3_rank.vectorize_features import (
    AS_FEATURE_VECTOR_NAMES, PATH_FEATURE_VECTOR_NAMES, vectorize_as_features,
    vectorize_path_features
)
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import DATA_DIR

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model definition
# ----------------------------------------------------------------------

_INPUT_SIZE_ASN = len(AS_FEATURE_VECTOR_NAMES)
_INPUT_SIZE_PATH = len(PATH_FEATURE_VECTOR_NAMES)

_RNN_NUM_LAYERS = 2
_RNN_HIDDEN_SIZE = 64

_PATH_MLP_OUT_SIZE = 64

class _RNN(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            _INPUT_SIZE_ASN, _RNN_HIDDEN_SIZE,
            num_layers=_RNN_NUM_LAYERS, batch_first=True
        )

        self.l1 = nn.Linear(_INPUT_SIZE_PATH, 16)
        self.l2 = nn.Linear(16, _PATH_MLP_OUT_SIZE)

        self.l_out1 = nn.Linear(_RNN_HIDDEN_SIZE + _PATH_MLP_OUT_SIZE, 16)
        self.l_out2 = nn.Linear(16, 1)

    def forward(
            self,
            x_as_level: torch.Tensor, # shape (len_path, amnt_of_as_features)
            x_path_level: torch.Tensor, # shape (amnt_of_path_features)
    ) -> torch.Tensor:
        hidden_0 = torch.zeros(_RNN_NUM_LAYERS, _RNN_HIDDEN_SIZE).to(_device)

        out_rnn, _ = self.rnn(x_as_level, hidden_0)
        # out shape: (seq_length, hidden_size)
        # - we want just the last hidden state -> we want shape (hidden_size)
        out_rnn = out_rnn[-1, :]

        out_mlp = self.l1(x_path_level)
        out_mlp = self.l2(out_mlp)

        out = self.l_out1(torch.concat((out_rnn, out_mlp)))
        out = self.l_out2(out)

        return out


_model = _RNN().to(_device)
_model_ready = False
_model_file_path = DATA_DIR / 'models' / 'rnn.pt'

_loss_fn = nn.BCEWithLogitsLoss()
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=0.001)
_epochs = 10

# Training & Evaluation
# ======================================================================

def _train(
        X_path_level: t.List[t.List[t.Union[float, int]]],
        X_as_level: t.List[t.List[t.List[t.Union[float, int]]]],
        y: t.List,
        eval_fn: t.Callable[[], t.Any],
):
    assert len(y) == len(X_path_level) and len(y) == len(X_as_level)

    _model_file_path.parent.mkdir(exist_ok=True, parents=True)
    if _model_file_path.exists():
        _LOG.info('Found model on disk, loading...')
        _model.load_state_dict(torch.load(_model_file_path))
        _LOG.info('Model loaded from disk')
        _model.eval()
        _model_ready = True
        return
    _LOG.info('Model not found on disk, initializing training')

    prg_i = 0
    prg_steps = 1000
    eval_steps = 10_000
    prg = Progress(int(len(y) * _epochs / prg_steps), 'train')
    losses: t.List[float] = []

    cancel = False

    sigint_orig_handler = signal.getsignal(signal.SIGINT)
    def sigint_handler(sig: int, frame: t.Optional[FrameType]):
        nonlocal cancel, sigint_orig_handler
        _LOG.warning('Training cancelled because SIGINT (Ctrl+C)')
        signal.signal(signal.SIGINT, sigint_orig_handler)
        cancel = True
    signal.signal(signal.SIGINT, sigint_handler)

    # Training Loop
    # ----------------------------------------------------------------------

    _model.train()
    for epoch in range(_epochs):
        for i in range(len(y)):
            # NOTE: This is *really* slow and cpu bound for now. The tutorial
            # that I used was assuming fixed length sequences, but we have
            # variable length sequences (AS paths). This complicates batching
            # because a tensor has to have fixed size dimensions.
            # Future potential optimization:
            # - torch.nn.utils.rnn.pack_sequence
            # - torch.nn.utils.rnn.unpack_sequence
            # - https://datascience.stackexchange.com/questions/120781

            X_path_level_el \
                          = torch.tensor(X_path_level[i], dtype=torch.float32, device=_device)
            X_as_level_el = torch.tensor(X_as_level[i], dtype=torch.float32, device=_device)
            y_el          = torch.tensor(y[i], dtype=torch.float32, device=_device)

            y_logits = _model(X_as_level_el, X_path_level_el).squeeze()
            loss = _loss_fn(y_logits, y_el)
            losses.append(loss.item())

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            prg_i += 1
            if prg_i % prg_steps == 0: prg.update(f'E{epoch}: loss {mean(losses)}')
            if prg_i % eval_steps == 0: eval_fn()

            if cancel: break

    signal.signal(signal.SIGINT, sigint_orig_handler)
    prg.complete()

    _LOG.info('Training Complete, saving model to disk...')
    torch.save(_model.state_dict(), _model_file_path)
    _LOG.info('Model saved to disk')
    _model_ready = True


def _evaluate(
        X_path_level: t.List[t.List[t.Union[float, int]]],
        X_as_level: t.List[t.List[t.List[t.Union[float, int]]]],
        y: t.List,
) -> t.List[float]:
    _LOG.info('Evaluating model...')
    with torch.inference_mode():
        prob_test: t.List[float] = []
        for i in range(len(y)):
            X_path_level_el \
                          = torch.tensor(X_path_level[i], dtype=torch.float32, device=_device)
            X_as_level_el = torch.tensor(X_as_level[i], dtype=torch.float32, device=_device)

            y_logits = _model(X_as_level_el, X_path_level_el).squeeze()
            prob_test.append(float(torch.sigmoid(y_logits)))

        f1 = binary_f1_score(torch.tensor(prob_test), torch.tensor(y))
        acc = binary_accuracy(torch.tensor(prob_test), torch.tensor(y))
        _LOG.info(f'F1={f1} ACC={acc}')

    return prob_test

_cpu_model = _RNN().to('cpu')

def predict_probs(paths: t.List[t.List[int]]) -> t.List[float]:
    global _model_ready, _device
    if not _model_ready:
        if not _model_file_path.exists(): _ready_model()
        _cpu_model.load_state_dict(torch.load(_model_file_path, map_location='cpu'))
        _device = 'cpu'
        _model_ready = True

    ret: t.List[float] = []

    for path in paths:
        as_features = [ vectorize_as_features(enrich_asn(asn)) for asn in path ]
        path_features = vectorize_path_features(enrich_path(path))
        y_logits = _cpu_model(
            torch.tensor(as_features, dtype=torch.float32),
            torch.tensor(path_features, dtype=torch.float32),
        ).squeeze()
        ret.append(float(torch.sigmoid(y_logits)))

    return ret


def _ready_model():
    test_split = 0.2
    dataset = make_dataset()
    _LOG.info(f'Got dataset with {len(dataset)} paths')

    _LOG.info('Preparing dataset...')

    random.shuffle(dataset)

    train_size = int(len(dataset) * (1 - test_split))
    _LOG.info(f'Train size = {train_size}')

    X_path_level = [ el['path_features'] for el in dataset ]
    X_as_level   = [ el['as_features'] for el in dataset ]
    y            = [ int(el['real']) for el in dataset ]

    X_path_level_train = X_path_level[:train_size]
    X_path_level_test  = X_path_level[train_size:]
    X_as_level_train   = X_as_level[:train_size]
    X_as_level_test    = X_as_level[train_size:]
    y_train            = y[:train_size]
    y_test             = y[train_size:]

    _LOG.info('Dataset prepared')

    def _evaluate_on_test():
        return _evaluate(X_path_level_test, X_as_level_test, y_test)

    _train(X_path_level_train, X_as_level_train, y_train, _evaluate_on_test)
    _evaluate_on_test()