import typing as t

_T = t.TypeVar('_T', bound=t.Hashable)

Graph = t.Dict[_T, t.Set[_T]]