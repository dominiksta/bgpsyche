import typing as t

WeightFunction = t.Callable[[int, int, t.Dict[str, t.Any]], float]
