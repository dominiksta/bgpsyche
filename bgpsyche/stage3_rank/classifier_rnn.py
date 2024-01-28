from datetime import datetime
import logging
from statistics import mean
from types import FrameType
import typing as t
import signal

import torch
from torch import nn
from torcheval.metrics.functional import binary_accuracy, binary_f1_score
from sklearn.model_selection import train_test_split
from bgpsyche.stage2_enrich.enrich import enrich_asn
from bgpsyche.stage3_rank.make_dataset import make_as_level_dataset
from bgpsyche.stage3_rank.real_life_eval import real_life_eval_model
from bgpsyche.stage3_rank.vectorize_features import vectorize_as_features
from bgpsyche.service.ext import routeviews
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import DATA_DIR

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model definition
# ----------------------------------------------------------------------

_INPUT_SIZE = 3 # see vectorize_as_features

_NUM_LAYERS = 2
_HIDDEN_SIZE = 64

class _RNN(nn.Module):
    def __init__(
            self,
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            _INPUT_SIZE, _HIDDEN_SIZE, num_layers=_NUM_LAYERS, batch_first=True
        )
        self.l1 = nn.Linear(_HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_0 = torch.zeros(_NUM_LAYERS, _HIDDEN_SIZE).to(_device)

        out, _ = self.rnn(x, hidden_0)
        # out shape: (seq_length, hidden_size)
        # - we want just the last hidden state
        # -> we want shape (hidden_size)
        out = out[-1, :]
        out = self.l1(out)
        return out


_model = _RNN().to(_device)

_loss_fn = nn.BCEWithLogitsLoss()
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=0.001)
_epochs = 1

# Training & Evaluation
# ======================================================================

def _train(
        X: t.List, y: t.List,
        eval_fn: t.Callable[[], t.Any],
):
    model_file_path = DATA_DIR / 'models' / 'rnn.pt'
    model_file_path.parent.mkdir(exist_ok=True, parents=True)
    if model_file_path.exists():
        _LOG.info('Found model on disk, loading...')
        _model.load_state_dict(torch.load(model_file_path))
        _LOG.info('Model loaded from disk')
        _model.eval()
        return
    _LOG.info('Model not found on disk, initializing training')

    prg_i = 0
    prg_steps = 1000
    eval_steps = 10_000
    prg = Progress(int(len(X) * _epochs / prg_steps), 'train')
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
        for X_train_el, y_train_el in zip(X, y):
            # NOTE: This is *really* slow and cpu bound for now. The tutorial
            # that I used was assuming fixed length sequences, but we have
            # variable length sequences (AS paths). This complicates batching
            # because a tensor has to have fixed size dimensions.
            # Future potential optimization:
            # - torch.nn.utils.rnn.pack_sequence
            # - torch.nn.utils.rnn.unpack_sequence
            # - https://datascience.stackexchange.com/questions/120781

            X_train_el = torch.tensor(X_train_el, dtype=torch.float32, device=_device)
            y_train_el = torch.tensor(y_train_el, dtype=torch.float32, device=_device)
            y_logits = _model(X_train_el).squeeze()
            loss = _loss_fn(y_logits, y_train_el)
            losses.append(loss.item())

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            prg_i += 1
            if prg_i % prg_steps == 0: prg.update(f'E{epoch}: loss {mean(losses)}')
            if prg_i % eval_steps == 0: eval_fn()

            if cancel: break

    prg.complete()

    _LOG.info('Training Complete, saving model to disk...')
    torch.save(_model.state_dict(), model_file_path)
    _LOG.info('Model saved to disk')


def _evaluate(X: t.List, y: t.List) -> t.List[float]:
    _LOG.info('Evaluating model...')
    with torch.inference_mode():
        prob_test: t.List[float] = []
        for _, (X_test_el, y_test_el) in enumerate(zip(X, y)):
            X_test_el = torch.tensor(X_test_el, dtype=torch.float32, device=_device)
            y_test_el = torch.tensor(y_test_el, dtype=torch.float32, device=_device)
            y_logits = _model(X_test_el).squeeze()
            # loss = _loss_fn(y_logits, y_test_el)
            prob_test.append(float(torch.sigmoid(y_logits)))


        f1 = binary_f1_score(torch.tensor(prob_test), torch.tensor(y))
        acc = binary_accuracy(torch.tensor(prob_test), torch.tensor(y))
        _LOG.info(f'F1={f1} ACC={acc}')

    return prob_test


def predict_probs(paths: t.List[t.List[int]]) -> t.List[float]:
    ret: t.List[float] = []

    for path in paths:
        as_features = [ vectorize_as_features(enrich_asn(asn)) for asn in path ]
        y_logits = _model(
            torch.tensor(as_features, dtype=torch.float32, device=_device)
        ).squeeze()
        ret.append(float(torch.sigmoid(y_logits)))

    return ret


def _main():
    dataset = make_as_level_dataset()
    X = [ el['as_features'] for el in dataset ]
    y = [ int(el['real']) for el in dataset ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, # 20% test, 80% train
        random_state=_RANDOM_SEED
    )

    def _evaluate_on_test(): return _evaluate(X_test, y_test)

    _train(X_train, y_train, _evaluate_on_test)

    routeviews_paths = list(
        meta['path'] for meta in
        routeviews.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00'),
            eliminate_path_prepending=True,
        )
    )

    real_life_eval_model(routeviews_paths, predict_probs)


if __name__ == '__main__': _main()