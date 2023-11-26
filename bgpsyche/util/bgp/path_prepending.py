import typing as t


_Path = t.TypeVar('_Path', bound=t.Union[t.List[str], t.List[int], str])
def eliminate_path_prepending(path: _Path) -> _Path:
    # TODO: unit test
    _path: t.List[t.Any] = t.cast(t.Any, path)
    if type(path) == str: _path = t.cast(str, path).split(' ')

    pos, _len = 0, len(_path)
    while _len > 1 and _len > pos + 1:
        # print(f'pos: {_path[pos]}')

        while _len > pos + 1 and _path[pos] == _path[pos+1]:
            # print(f'path before: {_path}')
            _path = _path[0:pos] + _path[pos+1:_len]
            # print(f'path after: {_path}')
            _len = len(_path)

        pos += 1
        _len = len(_path)

    if type(path) == str: return t.cast(_Path, ' '.join(_path))
    else: return t.cast(_Path, _path)