import typing as t
import logging
import random

from tqdm import tqdm
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torcheval.metrics.functional import (
    binary_accuracy, binary_recall, binary_precision, binary_f1_score
)
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.util.cancel_iter import cancel_iter
from .classifier_rnn import tensorboard_writer

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

_tsb = tensorboard_writer.add_scalar
_tens = lambda l: torch.tensor(l, dtype=torch.float32, device=device)


# model definition
# ----------------------------------------------------------------------

class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.rnn_as_level = nn.RNN(
            input_size=10, hidden_size=8, num_layers=8,
            batch_first=True,
            nonlinearity='tanh',
            # dropout?
            # bidirectional?
        )

        self.rnn_link_level = nn.RNN(
            input_size=4, hidden_size=8, num_layers=8,
            batch_first=True,
            nonlinearity='tanh',
            # dropout?
            # bidirectional?
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(8 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )


    def forward(self, X_as_level: torch.Tensor, X_link_level: torch.Tensor):
        # print(input.shape)
        # print(input[0])
        rnn_as_level_out, _ = self.rnn_as_level(X_as_level)
        # out shape: (batch_size, [padded] seq_length, hidden_size)
        # - we want just the last hidden state -> we want shape (batch_size, hidden_size)
        rnn_as_level_out = rnn_as_level_out[:, -1, :]

        rnn_link_level_out, _ = self.rnn_link_level(X_link_level)
        rnn_link_level_out = rnn_link_level_out[:, -1, :]

        # print(rnn_out.shape)
        # print(rnn_out[0])
        out = self.mlp_out(
            torch.concat([rnn_as_level_out, rnn_link_level_out], dim=1)
        )
        # print(out.shape)
        # print(out[0])
        return out


_model = _Model().to(device)

_loss_fn = nn.BCEWithLogitsLoss(
    # reduction?
    pos_weight=torch.tensor([5], device=device)
)

# alternative: try adam
_optimizer = torch.optim.Adam(params=_model.parameters(), lr=0.001)

_epochs = 100_000

# training loop
# ----------------------------------------------------------------------

def train(
        X_as_level: torch.Tensor,
        X_link_level: torch.Tensor,
        y: torch.Tensor
):

    # Train/Test split
    # ----------------------------------------------------------------------

    train_split = int(len(y) * 0.8)

    X_as_level_train   = X_as_level[:train_split]
    X_as_level_test    = X_as_level[train_split:]
    X_link_level_train = X_link_level[:train_split]
    X_link_level_test  = X_link_level[train_split:]
    y_train            = y[:train_split]
    y_test             = y[train_split:]


    for epoch in range(_epochs):
        # Training
        # ----------------------------------------------------------------------
        _model.train()

        # 1. Forward pass (model outputs raw logits)
        # print(X_train)
        y_logits = _model(
            X_as_level_train, X_link_level_train
        ).squeeze() # squeeze to remove extra `1` dimensions

        # 2. Calculate loss
        loss = _loss_fn(y_logits, y_train)  # using nn.BCEWithLogitsLoss works
                                            # with raw logits

        _optimizer.zero_grad() # 3. Optimizer zero grad
        loss.backward()        # 4. Loss backwards
        _optimizer.step()      # 5. Optimizer step

        # Logging Progress
        # ----------------------------------------------------------------------
        if epoch % 5 == 0:
            _model.eval()
            with torch.inference_mode():
                test_logits = _model(X_as_level_test, X_link_level_test).squeeze()

            loss_test = _loss_fn(test_logits, y_test)
            # prob_train=torch.sigmoid(y_logits)
            prob_test=torch.sigmoid(test_logits)
            # f1_train = binary_f1_score(prob_train, y_train)
            f1_test = binary_f1_score(prob_test, y_test)
            # acc_train = binary_accuracy(prob_train, y_train)
            acc_test = binary_accuracy(prob_test, y_test)
            # prec_train = binary_recision(prob_train, y_train)
            prec_test = binary_precision(prob_test, y_test)
            # prec_train = binary_recision(prob_train, y_train)
            rec_test = binary_recall(prob_test, y_test.bool())

            _LOG.info(
                f'Training Epoch: {epoch} | ' +
                f'Loss: {loss:.5f}, Test Loss: {loss_test:.5f} | ' +
                f'A: {acc_test:.3f}, P: {prec_test:.3f}, R: {rec_test:.3f}, ' +
                f'F1: {f1_test:.3f}'
            )
            _tsb('eval_synthetic_train/loss', loss, epoch)
            _tsb('eval_synthetic_test/loss', loss_test, epoch)
            _tsb('eval_synthetic_test/accuracy', acc_test, epoch)
            _tsb('eval_synthetic_test/precision', prec_test, epoch)
            _tsb('eval_synthetic_test/recall', rec_test, epoch)
            _tsb('eval_synthetic_test/f1', f1_test, epoch)
        else:
            _LOG.info(f'Training Epoch: {epoch} | Loss: {loss:.5f}')
            _tsb('eval_synthetic_train/loss', loss, epoch)


def _test():

    dataset = make_dataset()
    _LOG.info('Got Dataset')

    random.shuffle(dataset)

    X_as_level = pad_sequence(
        [
            _tens(
                [
                    [
                        # *ft_vec,
                        # ft_vec[0], # as_rank_cone
                        # ft_vec[1], # rirstat_born
                        # ft_vec[2], # rirstat_addr_count_v4
                        # ft_vec[3], # rirstat_addr_count_v6
                        ft_vec[4],  # 'category_unknown',
                        ft_vec[5],  # 'category_transit_access',
                        ft_vec[6],  # 'category_content',
                        ft_vec[7],  # 'category_enterprise',
                        ft_vec[8],  # 'category_educational_research',
                        ft_vec[9],  # 'category_non_profit',
                        ft_vec[10], # 'category_route_server',
                        ft_vec[11], # 'category_network_services',
                        ft_vec[12], # 'category_route_collector',
                        ft_vec[13], # 'category_government',
                        # 0,
                    ]
                    for ft_vec in el['as_features']
                ],
            )
            for el in cancel_iter(tqdm(dataset, 'make X_as_level tensor (can cancel)'))
        ],
        batch_first=True
    )

    X_link_level = pad_sequence(
        [
            _tens(
                [
                    [
                        # *ft_vec,
                        ft_vec[0], # rel_p2c
                        ft_vec[1], # rel_p2p
                        ft_vec[2], # rel_c2p
                        ft_vec[3], # rel_unknown
                        # 0,
                    ]
                    for ft_vec in el['link_features']
                ],
            )
            for el in tqdm(dataset[:len(X_as_level)], 'make X_link_level tensor')
        ],
        batch_first=True
    )

    y = _tens([
        int(p['real'])
        for p in tqdm(dataset[:len(X_as_level)], 'make y tensor')
    ])
    _LOG.info('Got training data set')

    train(X_as_level, X_link_level, y)



if __name__ == '__main__': _test()