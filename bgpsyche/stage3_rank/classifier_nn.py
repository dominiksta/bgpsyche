import logging
import typing as t

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from bgpsyche.stage3_rank.make_dataset import make_path_dataset
from torcheval.metrics.functional import (
    binary_accuracy, binary_recall, binary_precision, binary_f1_score
)

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# model definition
# ----------------------------------------------------------------------

_in_features = 5

_model = nn.Sequential(
    nn.Linear(in_features=_in_features, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=1),
    nn.ReLU(),
).to(device)

_loss_fn = nn.BCEWithLogitsLoss()

# alternative: try adam
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=0.001)

_epochs = 100_000

# training loop
# ----------------------------------------------------------------------

def train(X: torch.Tensor, y: torch.Tensor):

    # Train/Test split
    # ----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = t.cast(
        t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        train_test_split(
            X, y,
            test_size=0.2, # 20% test, 80% train
            random_state=_RANDOM_SEED
        )
    )

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(_epochs):
        # Training
        # ----------------------------------------------------------------------
        _model.train()

        # 1. Forward pass (model outputs raw logits)
        y_logits = _model(X_train).squeeze() # squeeze to remove extra `1`
                                             # dimensions

        # 2. Calculate loss
        loss = _loss_fn(y_logits, y_train)  # using nn.BCEWithLogitsLoss works
                                            # with raw logits

        _optimizer.zero_grad() # 3. Optimizer zero grad
        loss.backward()        # 4. Loss backwards
        _optimizer.step()      # 5. Optimizer step

        # Logging Progress
        # ----------------------------------------------------------------------
        if epoch % 10 == 0:
            _model.eval()
            with torch.inference_mode(): test_logits = _model(X_test).squeeze()

            loss_test = _loss_fn(test_logits, y_test)

            prob_train=torch.sigmoid(y_logits)
            prob_test=torch.sigmoid(test_logits)

            # f1_train = binary_f1_score(prob_train, y_train)
            f1_test = binary_f1_score(prob_test, y_test)

            # acc_train = binary_accuracy(prob_train, y_train)
            acc_test = binary_accuracy(prob_test, y_test)

            # prec_train = binary_precision(prob_train, y_train)
            prec_test = binary_precision(prob_test, y_test)

            _LOG.info(
                f'Training Epoch: {epoch} | ' +
                f'Loss: {loss:.5f}, Test Loss: {loss_test:.5f} | ' +
                f'A: {acc_test}, P: {prec_test}, F1: {f1_test}'
            )


def _test():

    dataset = make_path_dataset()
    X = torch.tensor(
        [ p['path_features'] for p in dataset ],
        dtype=torch.float32, device=device
    )
    y = torch.tensor(
        [ int(p['real']) for p in dataset ],
        dtype=torch.float32, device=device
    )
    _LOG.info('Got training data set')

    
    print(X.shape)
    print(y.shape)
    print(y[:10])

    train(X, y)

    

if __name__ == '__main__': _test()