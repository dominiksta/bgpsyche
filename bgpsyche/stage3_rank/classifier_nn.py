from itertools import pairwise
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
from bgpsyche.util.cancel_iter import cancel_iter
from bgpsyche.util.const import DATA_DIR
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_asn, enrich_link, enrich_path
from bgpsyche.stage3_rank.make_dataset import DatasetEl, make_dataset
from bgpsyche.stage3_rank.vectorize_features import (
    AS_FEATURE_VECTOR_NAMES, LINK_FEATURE_VECTOR_NAMES, PATH_FEATURE_VECTOR_NAMES, vectorize_as_features, vectorize_link_features, vectorize_path_features
)
from bgpsyche.stage3_rank.tensorboard import tensorboard_writer

_LOG = logging.getLogger(__name__)

_RANDOM_SEED = 1337
torch.manual_seed(_RANDOM_SEED)
if torch.cuda.is_available: torch.cuda.manual_seed(_RANDOM_SEED)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# _device = 'cpu'

_tsb = tensorboard_writer.add_scalar
_tens = lambda l: torch.tensor(l, dtype=torch.float32, device=_device)


# model definition
# ----------------------------------------------------------------------

class _Model(nn.Module):
    def __init__(
            self,
            input_size_link_level: int,
            input_size_as_level: int,
            input_size_path_level: int,
    ) -> None:
        super().__init__()
        self._input_size_link_level = input_size_link_level
        self._input_size_as_level   = input_size_as_level
        self._input_size_path_level = input_size_path_level

        self.rnn_as_level = nn.RNN(
            input_size=self._input_size_as_level,
            hidden_size=8, num_layers=8,
            batch_first=True,
            nonlinearity='tanh',
            # dropout?
            # bidirectional?
        )

        self.rnn_link_level = nn.RNN(
            input_size=self._input_size_link_level,
            hidden_size=8, num_layers=8,
            batch_first=True,
            nonlinearity='tanh',
            # dropout?
            # bidirectional?
        )

        self.mlp_path_level = nn.Sequential(
            nn.Linear(self._input_size_path_level, 8),
            nn.ReLU(),
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(8 + 8 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )


    def forward(
            self,
            X_as_level: torch.Tensor,
            X_link_level: torch.Tensor,
            X_path_level: torch.Tensor,
    ):
        # print(input.shape)
        # print(input[0])
        rnn_as_level_out, _ = self.rnn_as_level(X_as_level)
        # out shape: (batch_size, [padded] seq_length, hidden_size)
        # - we want just the last hidden state -> we want shape (batch_size, hidden_size)
        rnn_as_level_out = rnn_as_level_out[:, -1, :]

        rnn_link_level_out, _ = self.rnn_link_level(X_link_level)
        rnn_link_level_out = rnn_link_level_out[:, -1, :]

        mlp_path_level_out = self.mlp_path_level(X_path_level)

        # print(rnn_out.shape)
        # print(rnn_out[0])
        out = self.mlp_out(
            torch.concat([
                rnn_as_level_out, rnn_link_level_out, mlp_path_level_out
            ], dim=1)
        )
        # print(out.shape)
        # print(out[0])
        return out


_epochs = 100_000


class _Dataset(t.TypedDict):
    X_as_level   : torch.Tensor
    X_link_level : torch.Tensor
    X_path_level : torch.Tensor
    y            : torch.Tensor


def train(dataset: _Dataset) -> t.Dict[str, t.Any]: # return state_dict

    # model initialization
    # ----------------------------------------------------------------------

    model = _Model(
        input_size_as_level   = len(dataset['X_as_level'][0][0]),
        input_size_link_level = len(dataset['X_link_level'][0][0]),
        input_size_path_level = len(dataset['X_path_level'][0]),
    ).to(_device)

    loss_fn = nn.BCEWithLogitsLoss(
        # reduction?
        # HACK: this should probably not be hard coded
        pos_weight=torch.tensor([5], device=_device)
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


    # train/test split
    # ----------------------------------------------------------------------

    train_split = int(len(dataset['y']) * 0.8)

    X_as_level_train   = dataset['X_as_level'][:train_split]
    X_as_level_test    = dataset['X_as_level'][train_split:]
    X_link_level_train = dataset['X_link_level'][:train_split]
    X_link_level_test  = dataset['X_link_level'][train_split:]
    X_path_level_train = dataset['X_path_level'][:train_split]
    X_path_level_test  = dataset['X_path_level'][train_split:]
    y_train            = dataset['y'][:train_split]
    y_test             = dataset['y'][train_split:]


    for epoch in cancel_iter(range(_epochs), name='model training'):

        # training
        # ----------------------------------------------------------------------

        model.train()

        # 1. Forward pass (model outputs raw logits)
        # print(X_train)
        y_logits = model(
            X_as_level_train, X_link_level_train, X_path_level_train,
        ).squeeze() # squeeze to remove extra `1` dimensions

        # 2. Calculate loss
        loss = loss_fn(y_logits, y_train)  # using nn.BCEWithLogitsLoss works
                                            # with raw logits

        optimizer.zero_grad() # 3. Optimizer zero grad
        loss.backward()        # 4. Loss backwards
        optimizer.step()      # 5. Optimizer step

        # logging & visualizing progress
        # ----------------------------------------------------------------------

        if epoch % 5 == 0:
            model.eval()
            with torch.inference_mode():
                test_logits = model(
                    X_as_level_test, X_link_level_test, X_path_level_test
                ).squeeze()

            loss_test = loss_fn(test_logits, y_test)
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

    return model.state_dict()


def _load_dataset() -> _Dataset:

    dataset = make_dataset()
    _LOG.info('Dataset construction finished, now loading as tensors...')

    random.shuffle(dataset)

    for el in tqdm(dataset, 'Applying runtime dataset transforms'):
        for transform in _DATASET_RUNTIME_INPUT_TRANSFORMERS: transform(el)

    X_as_level = pad_sequence(
        [
            _tens(el['as_features'])
            for el in cancel_iter(tqdm(dataset, 'make X_as_level tensor (can cancel)'))
        ],
        batch_first=True
    )

    X_link_level = pad_sequence(
        [
            _tens(el['link_features'])
            for el in tqdm(dataset[:len(X_as_level)], 'make X_link_level tensor')
        ],
        batch_first=True
    )

    X_path_level = _tens(
        [
            el['path_features']
            for el in tqdm(dataset[:len(X_as_level)], 'make X_path_level tensor')
        ]
    )

    y = _tens([
        int(p['real'])
        for p in tqdm(dataset[:len(X_as_level)], 'make y tensor')
    ])

    return {
        'X_as_level': X_as_level,
        'X_link_level': X_link_level,
        'X_path_level': X_path_level,
        'y': y
    }


# runtime dataset transformation (just for faster experimentation)
# ----------------------------------------------------------------------

class _DatasetElInput(t.TypedDict):
    as_features   : t.List[t.List[t.Union[float, int]]]
    link_features : t.List[t.List[t.Union[float, int]]]
    path_features : t.List[t.Union[float, int]]


_DatasetRuntimeInputTransformer = t.Callable[[_DatasetElInput], None]


def _dataset_transform_pick_features(el: _DatasetElInput):
    el['as_features'] = [
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
        ] for ft_vec in el['as_features']
    ]
    el['link_features'] = [
        [
            # *ft_vec,
            ft_vec[0], # rel_p2c
            ft_vec[1], # rel_p2p
            ft_vec[2], # rel_c2p
            ft_vec[3], # rel_unknown
            # 0,
        ] for ft_vec in el['link_features']
    ]
    el['path_features'] = [
        el['path_features'][0], # length
        # el['path_features'][1], # is_valley_free
    ]


_DATASET_RUNTIME_INPUT_TRANSFORMERS: t.List[_DatasetRuntimeInputTransformer] = [
    _dataset_transform_pick_features
]


def _ready_model_default_train(
        retrain = False,
        device = 'cpu',
) -> _Model:

    file_path = DATA_DIR / 'models' / 'bgpsyche.pt'
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        _LOG.info('Model not found on disk')
        retrain = True

    if retrain:
        _LOG.info('Initializing training')
        dataset = _load_dataset()
        state_dict = train(dataset)
        _LOG.info('Saving model to disk')
        torch.save(state_dict, file_path)

    model = _Model(
        input_size_as_level   = len(dataset['X_as_level'][0][0]),
        input_size_link_level = len(dataset['X_link_level'][0][0]),
        input_size_path_level = len(dataset['X_path_level'][0]),
    ).to(device)
    model.load_state_dict(torch.load(file_path, map_location=device))

    _LOG.info('Model is ready to make predictions')

    return model


def make_prediction_function():

    model = _ready_model_default_train(retrain=True, device='cpu')
    model.eval()
    _tens = lambda l: torch.tensor(l, dtype=torch.float32, device='cpu')

    def predict_probs(paths: t.List[t.List[int]]) -> t.List[float]:
        out: t.List[float] = []

        # for *some reason*, we cannot first get all features for all paths and then
        # pack them into one set of tensors. pytorch just gets completely stuck
        # then. i even attempted implementing my own pack_sequence, only to discover
        # that the simple act of calling torch.tensor() on a 3-dimensional list of
        # shape ~ 8*1000*10 will cause the bug. this may have to do something with
        # multiprocessing, maybe not. it does not even use any cpu, it just sits
        # there idle waiting for better days to come.

        for path in paths:
            input: _DatasetElInput = {
                'as_features': [
                    vectorize_as_features(enrich_asn(asn))
                    for asn in path
                ],
                'link_features': [
                    vectorize_link_features(enrich_link(source, sink))
                    for source, sink in pairwise(path)
                ],
                'path_features': vectorize_path_features(enrich_path(path)),

            }
            for transform in _DATASET_RUNTIME_INPUT_TRANSFORMERS: transform(input)

            X_as_level   = _tens([input['as_features']])
            X_link_level = _tens([input['link_features']])
            X_path_level = _tens([input['path_features']])

            with torch.inference_mode():
                y_logits = model(X_as_level, X_link_level, X_path_level)

            out.append(float(torch.sigmoid(y_logits)))

        return out

    return predict_probs


if __name__ == '__main__':

    predict_probs = make_prediction_function()

    candidates = get_path_candidates(3320, 8075)[:10] # dtag -> microsoft
    probs = predict_probs(candidates)

    for i in range(len(candidates)):
        _LOG.info(f'Prob: {probs[i]}, Route: {candidates[i]}')
