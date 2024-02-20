import typing as t
from itertools import islice

_T = t.TypeVar('_T')

def iter_batched(
        iterable: t.Iterable[_T],
        batch_size: int,
) -> t.Iterator[t.List[_T]]:
    # from https://docs.python.org/3/library/itertools.html#itertools.batched
    it = iter(iterable)
    while batch := tuple(islice(it, batch_size)):
        yield t.cast(t.Any, batch)
