import typing as t

def one_hot(
        names: t.List[str],
        optional: bool,
) -> t.Callable[[t.Optional[str]], t.List[t.Literal[0, 1]]]:
    names_len = len(names)
    if optional: names_len += 1

    def _one_hot(name: t.Optional[str]) -> t.List[t.Literal[0, 1]]:
        assert name is not None or optional
        i = names.index(name) if name is not None else names_len - 1
        ret: t.List[t.Literal[0, 1]] = [0] * names_len
        ret[i] = 1
        return ret

    return _one_hot


def scale_zero_to_one_linear(
        val: t.Optional[float],
        val_min: float, val_max: float,
        allow_outlier = True,
) -> float:
    if val is None: return 0

    if allow_outlier: val = min(val_max, max(val_min, val))
    else: assert val >= val_min and val <= val_max

    return (val - val_min) / (val_max - val_min)