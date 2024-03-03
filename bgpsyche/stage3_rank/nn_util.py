import random
import typing as t
from copy import deepcopy
from itertools import islice

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
