from copy import deepcopy
from datetime import datetime
import logging
from itertools import pairwise
from math import ceil
from statistics import mean
from types import FrameType
import typing as t
import signal
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall
from bgpsyche.stage2_enrich.enrich import enrich_asn, enrich_link, enrich_path
from bgpsyche.stage3_rank.make_dataset import DatasetEl, make_dataset
from bgpsyche.stage3_rank.nn_util import iter_batched
from bgpsyche.stage3_rank.vectorize_features import (
    AS_FEATURE_VECTOR_NAMES, LINK_FEATURE_VECTOR_NAMES, PATH_FEATURE_VECTOR_NAMES,
    vectorize_as_features, vectorize_link_features, vectorize_path_features
)
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import DATA_DIR

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_tensorboard_dir = DATA_DIR / 'tensorboard'
tensorboard_writer = SummaryWriter(
    _tensorboard_dir /
    f'{datetime.now().strftime("%m.%d_%H.%M.%S")}_{input("Name run: ")}'
)

# model definition
# ----------------------------------------------------------------------

_INPUT_SIZE_LINK = len(LINK_FEATURE_VECTOR_NAMES)
_INPUT_SIZE_ASN = len(AS_FEATURE_VECTOR_NAMES)
_INPUT_SIZE_PATH = len(PATH_FEATURE_VECTOR_NAMES)

_RNN_AS_LEVEL_NUM_LAYERS = 2
_RNN_AS_LEVEL_HIDDEN_SIZE = 64

_RNN_LINK_LEVEL_NUM_LAYERS = 2
_RNN_LINK_LEVEL_HIDDEN_SIZE = 64

_PATH_MLP_OUT_SIZE = 64

class _RNN(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

        self.rnn_as_level = nn.RNN(
            _INPUT_SIZE_ASN, _RNN_AS_LEVEL_HIDDEN_SIZE,
            num_layers=_RNN_AS_LEVEL_NUM_LAYERS, batch_first=True
        )
        self.rnn_link_level = nn.RNN(
            _INPUT_SIZE_LINK, _RNN_LINK_LEVEL_HIDDEN_SIZE,
            num_layers=_RNN_LINK_LEVEL_NUM_LAYERS, batch_first=True
        )

        self.l1 = nn.Linear(_INPUT_SIZE_PATH, 16)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(16, _PATH_MLP_OUT_SIZE)
        self.tanh2 = nn.Tanh()

        self.l_out1 = nn.Linear(
            _RNN_AS_LEVEL_HIDDEN_SIZE
            + _RNN_LINK_LEVEL_HIDDEN_SIZE
            + _PATH_MLP_OUT_SIZE,
            64
        )
        self.tanh_out1 = nn.Tanh()
        self.l_out2 = nn.Linear(64, 16)
        self.tanh_out2 = nn.Tanh()
        self.l_out3 = nn.Linear(16, 1)

    def forward(
            self,
            # shape [packed] (batch_size, len_path, amnt_of_as_features)
            x_as_level: torch.Tensor,
            # shape [packed] (batch_size, len_path-1, amnt_of_link_features)
            x_link_level: torch.Tensor,
            # shape [packed] (batch_size, amnt_of_path_features)
            x_path_level: torch.Tensor,
    ) -> torch.Tensor:
        # note: torch.nn.RNN uses the tanh activation function internally for
        # non-linearity by default

        out_rnn_as_level, _ = self.rnn_as_level(x_as_level)
        # out shape: (batch_size, [padded] seq_length, hidden_size)
        # - we want just the last hidden state -> we want shape (batch_size, hidden_size)
        out_rnn_as_level = \
            pad_packed_sequence(out_rnn_as_level, batch_first=True)[0][:, -1, :]

        out_rnn_link_level, _ = self.rnn_link_level(x_link_level)
        out_rnn_link_level = \
            pad_packed_sequence(out_rnn_link_level, batch_first=True)[0][:, -1, :]

        out_mlp = self.l1(x_path_level)
        out_mlp = self.tanh1(out_mlp)
        out_mlp = self.l2(out_mlp)
        out_mlp = self.tanh2(out_mlp)

        out = self.l_out1(torch.concat((
            out_rnn_as_level, out_rnn_link_level, out_mlp
        ), dim=1))
        out = self.tanh_out1(out)
        out = self.l_out2(out)
        out = self.tanh_out2(out)
        out = self.l_out3(out)
        # sigmoid is implied by BCEWithLogitsLoss

        return out


_model = _RNN().to(_device)
_model_ready = False
_model_file_path = DATA_DIR / 'models' / 'rnn.pt'

_loss_fn = nn.BCEWithLogitsLoss()
_epochs = 10
_learning_rate = 0.001
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=_learning_rate)

# Training & Evaluation
# ======================================================================

_BATCH_SIZE = 10

def _train(
        X_path_level : t.List[t.List[t.Union[float, int]]],
        X_as_level   : t.List[t.List[t.List[t.Union[float, int]]]],
        X_link_level : t.List[t.List[t.List[t.Union[float, int]]]],
        y: t.List,
        eval_fn: t.Callable[[int], t.Any],
):
    assert (
        len(y) == len(X_path_level) and
        len(y) == len(X_as_level) and
        len(y) == len(X_link_level)
    )

    _model_file_path.parent.mkdir(exist_ok=True, parents=True)
    _LOG.info('Model not found on disk, initializing training')

    prg_i = 0
    prg_steps = ceil(1000 / _BATCH_SIZE)
    prg_step_real = lambda: ceil((prg_i * _BATCH_SIZE) / 1000)
    eval_steps = int(10_000 / _BATCH_SIZE)
    prg = Progress(int(len(y) * _epochs / prg_steps / _BATCH_SIZE), 'train')
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
        iter_X_path_level = iter_batched(X_path_level, _BATCH_SIZE)
        iter_X_as_level   = iter_batched(X_as_level, _BATCH_SIZE)
        iter_X_link_level = iter_batched(X_link_level, _BATCH_SIZE)
        iter_y            = iter_batched(y, _BATCH_SIZE)

        for X_path_level_batch, X_as_level_batch, X_link_level_batch, y_batch \
                in zip(
                    iter_X_path_level, iter_X_as_level, iter_X_link_level, iter_y
                ):
            X_path_level_batch = torch.tensor(
                X_path_level_batch, dtype=torch.float32
            ).to(_device)
            X_as_level_batch = pack_sequence([
                torch.tensor(el, dtype=torch.float32) for el in X_as_level_batch
            ], enforce_sorted=False).to(_device)
            X_link_level_batch = pack_sequence([
                torch.tensor(el, dtype=torch.float32) for el in X_link_level_batch
            ], enforce_sorted=False).to(_device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=_device)

            y_logits = _model(
                X_as_level_batch, X_link_level_batch, X_path_level_batch
            ).squeeze()
            loss = _loss_fn(y_logits, y_batch)
            losses.append(loss.item())

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            prg_i += 1
            if prg_i % prg_steps == 0:
                tensorboard_writer.add_scalar(
                    'eval_synthetic_train/loss',
                    mean(losses[-prg_steps:]),
                    prg_step_real()
                )
                prg.update(f'E{epoch}: loss {mean(losses[-prg_steps:])}')

            if prg_i % eval_steps == 0: eval_fn(prg_step_real())

            if cancel: break

        if cancel: break

    tensorboard_writer.add_scalar(
        'eval_synthetic_train/loss',
        mean(losses[-prg_steps:]),
        prg_step_real()
    )
    eval_fn(prg_step_real())

    signal.signal(signal.SIGINT, sigint_orig_handler)
    prg.complete()

    _LOG.info('Training Complete, saving model to disk...')
    torch.save(_model.state_dict(), _model_file_path)
    _LOG.info('Model saved to disk')
    _model_ready = True


def _evaluate(
        prg_i: int,
        dataset_test: t.List[DatasetEl],
        dataset_train: t.List[DatasetEl],
        max_eval_paths_n = 5_000,
):
    _LOG.info('Evaluating model...')

    def take(l):
        copy = deepcopy(l)
        random.shuffle(copy)
        return copy[:max_eval_paths_n]

    dataset_test  = take(dataset_test)
    dataset_train = take(dataset_train)

    X_path_level_test  = [ el['path_features'] for el in dataset_test ]
    X_as_level_test    = [ el['as_features'] for el in dataset_test ]
    X_link_level_test  = [ el['link_features'] for el in dataset_test ]
    y_test             = [ int(el['real']) for el in dataset_test ]
    X_path_level_train = [ el['path_features'] for el in dataset_train ]
    X_as_level_train   = [ el['as_features'] for el in dataset_train ]
    X_link_level_train = [ el['link_features'] for el in dataset_train ]
    y_train            = [ int(el['real']) for el in dataset_train ]

    def pack(l: t.List):
        return pack_sequence([
            torch.tensor(el, dtype=torch.float32) for el in l
        ], enforce_sorted=False).to(_device)

    def show(name: str, probs: t.List[float], y: t.List) -> None:
        f1   = binary_f1_score (torch.tensor(probs), torch.tensor(y))
        acc  = binary_accuracy (torch.tensor(probs), torch.tensor(y))
        prec = binary_precision(torch.tensor(probs), torch.tensor(y))
        rec  = binary_recall   (torch.tensor(probs), torch.tensor(y))
        tensorboard_writer.add_scalar(f'eval_synthetic_{name}/f1', f1, prg_i)
        tensorboard_writer.add_scalar(f'eval_synthetic_{name}/accuracy', acc, prg_i)
        tensorboard_writer.add_scalar(f'eval_synthetic_{name}/precision', prec, prg_i)
        tensorboard_writer.add_scalar(f'eval_synthetic_{name}/recall', rec, prg_i)
        _LOG.info(f'Eval {name.upper()} F1={f1} ACC={acc} PREC={prec} REC={rec}')

    def itb(l: t.List) -> t.Iterator[t.List]:
        return iter_batched(l, _BATCH_SIZE)

    with torch.inference_mode():
        prob_test: t.List[float] = []

        for X_path_level_batch, X_as_level_batch, X_link_level_batch in zip(
                itb(X_path_level_test), itb(X_as_level_test), itb(X_link_level_test)
        ):
            y_logits = _model(
                pack(X_as_level_batch), pack(X_link_level_batch),
                torch.tensor(X_path_level_batch, dtype=torch.float32, device=_device),
            ).squeeze()
            prob_test += [ float(n) for n in torch.sigmoid(y_logits) ]

        show('test', prob_test, y_test)

        prob_train: t.List[float] = []

        for X_path_level_batch, X_as_level_batch, X_link_level_batch in zip(
                itb(X_path_level_train), itb(X_as_level_train), itb(X_link_level_train)
        ):
            y_logits = _model(
                pack(X_as_level_batch), pack(X_link_level_batch),
                torch.tensor(X_path_level_batch, dtype=torch.float32, device=_device),
            ).squeeze()
            prob_train += [ float(n) for n in torch.sigmoid(y_logits) ]

        show('train', prob_train, y_train)


_cpu_model = _RNN().to('cpu')

def predict_probs(
        paths: t.List[t.List[int]],
        retrain = True,
) -> t.List[float]:
    global _model_ready, _device
    if not _model_ready:
        if not _model_file_path.exists() or retrain: ready_model()
        _cpu_model.load_state_dict(torch.load(_model_file_path, map_location='cpu'))
        _device = 'cpu'
        _model_ready = True
        _cpu_model.eval()

    X_as_level = pack_sequence([
        torch.tensor(
            [ vectorize_as_features(enrich_asn(asn)) for asn in path ],
            dtype=torch.float32
        )
        for path in paths
    ], enforce_sorted=False)
    X_link_level = pack_sequence([
        torch.tensor(
            [ vectorize_link_features(enrich_link(source, sink))
              for source, sink in pairwise(path) ],
            dtype=torch.float32
        )
        for path in paths
    ], enforce_sorted=False)
    X_path_level = torch.tensor(
        [ vectorize_path_features(enrich_path(path)) for path in paths ],
        dtype=torch.float32
    )

    with torch.inference_mode():
        y_logits = _cpu_model(X_as_level, X_link_level, X_path_level)

    return [ float(el) for el in torch.sigmoid(y_logits) ]


def ready_model():
    test_split = 0.2
    dataset = make_dataset()
    _LOG.info(f'Got dataset with {len(dataset)} paths')

    _LOG.info('Preparing dataset...')

    random.shuffle(dataset)

    train_size = int(len(dataset) * (1 - test_split))
    _LOG.info(f'Train size = {train_size}')

    dataset_train = dataset[:train_size]
    dataset_test  = dataset[train_size:]

    X_path_level_train = [ el['path_features'] for el in dataset_train ]
    X_as_level_train   = [ el['as_features'] for el in dataset_train ]
    X_link_level_train = [ el['link_features'] for el in dataset_train ]
    y_train            = [ int(el['real']) for el in dataset_train ]

    _LOG.info('Dataset prepared')

    tensorboard_writer.add_text('hyperparameters', f"""
```
Training:
----------------------------------------------------------------------

Dataset Size  = {len(dataset):_}
Epochs        = {_epochs}
Batch Size    = {_BATCH_SIZE}
Learning Rate = {_learning_rate}

Network Definition:
----------------------------------------------------------------------

_RNN_AS_LEVEL_NUM_LAYERS    = {_RNN_AS_LEVEL_NUM_LAYERS}
_RNN_AS_LEVEL_HIDDEN_SIZE   = {_RNN_AS_LEVEL_HIDDEN_SIZE}
_RNN_LINK_LEVEL_NUM_LAYERS  = {_RNN_LINK_LEVEL_NUM_LAYERS}
_RNN_LINK_LEVEL_HIDDEN_SIZE = {_RNN_LINK_LEVEL_HIDDEN_SIZE}
_PATH_MLP_OUT_SIZE          = {_PATH_MLP_OUT_SIZE}
```
""")

    tensorboard_writer.flush()

    def _evaluate_on_test(prg_i: int):
        _evaluate(prg_i, dataset_test, dataset_train)

    _train(
        X_path_level_train,
        X_as_level_train,
        X_link_level_train,
        y_train,
        _evaluate_on_test
    )