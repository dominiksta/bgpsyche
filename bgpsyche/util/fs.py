import typing as t
from pathlib import Path
import os
import json
import gzip
import logging

from bgpsyche.util.const import DATA_DIR

_T = t.TypeVar("_T")
_LOG = logging.getLogger(__name__)


def shorter_readable_path(path: Path) -> str:
    parts = list(path.parts)
    if len(parts) > 4:
        for i in range(0, len(parts) - 4):
            parts[i] = parts[i][0]
    return '/' + '/'.join(parts[1:])


def json_file_cache(name: str, getter: t.Callable[[], _T]) -> _T:
    file = DATA_DIR / 'json_cache' / f'{name}.json'
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        _LOG.debug(f'JSON File Cache Hit: {name}')
        with open(file, 'r') as f: return json.loads(f.read())
    else:
        _LOG.info(f'JSON File Cache Miss: {name}')
        try:
            data = getter()
            with open(file, 'w') as f: f.write(json.dumps(data, indent=2))
            return data
        except:
            if file.exists(): os.remove(file)
            raise


def gzip_file(source: Path, dest: t.Optional[Path] = None):
    _dest = Path(source.parent.name + (source.name + '.gz')) \
        if dest is None else dest
    _LOG.info(
        f'gzip: {shorter_readable_path(source)} -> {shorter_readable_path(_dest)}'
    )
    with open(source, 'rb') as f_in, gzip.open(_dest, 'wb') as f_out:
        f_out.writelines(f_in)