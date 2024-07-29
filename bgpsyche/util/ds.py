import typing as t

_T = t.TypeVar('_T')

def str_to_set(str_in: str) -> set:
    without_brackets = str_in[1:-1]
    parts = list(map(lambda el: el.strip(), without_brackets.split(',')))
    ret: list[t.Any] = []
    if not (len(parts) == 1 and parts[0] == ''):
        for part in parts:
            if part.startswith('\'') or part.startswith('"'):
                ret.append(part[1:-1])
            else:
                ret.append(int(part))
    return set(ret)