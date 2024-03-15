import logging
import random
from pprint import pformat
import typing as t
from copy import deepcopy
from itertools import islice

import torch

_T = t.TypeVar('_T')

def iter_batched(
        items: t.List[_T],
        batch_size: int,
        shuffle = False,
        stop_after: t.Union[t.Literal['inf'], int] = 'inf',
) -> t.Iterator[t.List[_T]]:
    # from https://docs.python.org/3/library/itertools.html#itertools.batched
    items_local = deepcopy(items) if shuffle else items
    if shuffle: random.shuffle(items_local)
    it = iter(items)
    i = 0
    while batch := tuple(islice(it, batch_size)):
        i += batch_size
        if stop_after != 'inf' and i >= stop_after: break
        yield t.cast(t.Any, batch)


class EarlyStopping:
    # credits: https://github.com/jeffheaton/app_deep_learning

    _LOG = logging.getLogger('early_stop')

    def __init__(
            self,
            # how much of a change do we want to ignore
            min_delta: float = 0,
            # how many insignificant changes to loss do we want to allow before
            # stopping
            patience: int = 5,
            # wether the model should be restored to the state it was in when
            # the loss was lowest
            restore_best_weights: bool = True
    ):
        self._patience: int = patience
        self._min_delta: float= min_delta
        self._restore_best_weights: bool = restore_best_weights
        self._best_model: t.Mapping[str, t.Any] = {}
        self._best_loss: t.Optional[float] = None
        self._counter: int = 0

    @property
    def best_loss(self): return self._best_loss

    def __repr__(self) -> str:
        return pformat({
            'best_loss': self.best_loss,
            'patience': self._patience, 'min_delta': self._min_delta,
        })

    def __call__(self, model: torch.nn.Module, val_loss: float) -> bool:
        if self._best_loss is None:
            self._best_loss = val_loss
            self._best_model = deepcopy(model.state_dict())
        elif self._best_loss - val_loss >= self._min_delta:
            self._best_model = deepcopy(model.state_dict())
            self._best_loss = val_loss
            self._counter = 0
            self.__class__._LOG.debug(
                f'Improvement found, counter reset to {self._counter}'
            )
        else:
            self._counter += 1
            self.__class__._LOG.info(
                f'No improvement in the last {self._counter} epochs'
            )
            if self._counter >= self._patience:
                self.__class__._LOG.warning(
                    f'Early stopping triggered after {self._counter} epochs.'
                )
                self.__class__._LOG.warning(self)
                if self._restore_best_weights:
                    model.load_state_dict(self._best_model)
                return True
        return False