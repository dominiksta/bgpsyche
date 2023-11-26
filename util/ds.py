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

# dedent
# ----------------------------------------------------------------------

def dedent(margin: int, str_in: str, strip_lf: bool = True) -> str:
    out = ''
    for line in str_in.split('\n'):
        if line.startswith(' ' * margin): out += line[margin:]
        else: out += line
        out += '\n'
    if strip_lf: out = out.strip('\n')
    return out

def indent(margin: int, str_in: str, strip_lf: bool = False) -> str:
    out = str_in.split('\n')[0] + '\n'
    for line in str_in.split('\n')[1:]:
        out += ( ' ' * margin ) + line + '\n'
    if strip_lf: out = out.strip('\n')
    return out



def max_dict_keys(d: t.Dict[_T, int]) -> t.List[_T]:
    """
    Return the key in the given dict which points to the highest integer value.

    Example:

    max_dict_key({
      'foo': 7,
      'bar': 3,
      'baz': 9,
      'baz2': 9,
    })
    => ['baz', 'baz2']
    """
    max_keys: t.List[_T] = []
    max_val = 0
    for key, value in d.items():
        if max_val <= value:
            max_keys.append(key)
            max_val = value
    max_keys.sort() # type: ignore
    return max_keys