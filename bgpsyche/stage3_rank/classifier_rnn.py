import logging
from statistics import mean
import typing as t

import torch
from torch import nn
from torcheval.metrics.functional import binary_accuracy, binary_f1_score
from sklearn.model_selection import train_test_split
from bgpsyche.stage3_rank.make_dataset import make_as_level_dataset
from bgpsyche.util.benchmark import Progress

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
_epochs = 10_000

# Training & Evaluation
# ======================================================================

def _train():
    dataset = make_as_level_dataset()
    X = [ el['as_features'] for el in dataset ]
    y = [ int(el['real']) for el in dataset ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, # 20% test, 80% train
        random_state=_RANDOM_SEED
    )

    prg_i = 0
    prg_steps = 1000
    prg = Progress(int(len(dataset) * _epochs / prg_steps), 'train')
    losses: t.List[float] = []

    # Evaluation
    # ----------------------------------------------------------------------

    def eval():
        with torch.inference_mode():
            prob_test: t.List[float] = []
            for _, (X_test_el, y_test_el) in enumerate(zip(X_test, y_test)):
                X_test_el = torch.tensor(X_test_el, dtype=torch.float32, device=_device)
                y_test_el = torch.tensor(y_test_el, dtype=torch.float32, device=_device)
                y_logits = _model(X_test_el).squeeze()
                # loss = _loss_fn(y_logits, y_test_el)
                prob_test.append(float(torch.sigmoid(y_logits)))


            f1 = binary_f1_score(torch.tensor(prob_test), torch.tensor(y_test))
            acc = binary_accuracy(torch.tensor(prob_test), torch.tensor(y_test))
            _LOG.info(f'F1={f1} ACC={acc}')

    # Training
    # ----------------------------------------------------------------------

    _model.train()
    for epoch in range(_epochs):
        for X_train_el, y_train_el in zip(X_train, y_train):
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
            if prg_i % prg_steps == 0:
                prg.update(f'E{epoch}: loss {mean(losses)}')
                eval()

    prg.complete()


if __name__ == '__main__': _train()