import json
import logging
from pathlib import Path
from sys import stdout
import typing as t
import subprocess
from tempfile import NamedTemporaryFile

import bgpsyche.logging_config
from bgpsyche.util.const import DATA_DIR, HERE

"""
Here be dragons.
"""

_LOG = logging.getLogger(__name__)

_REPO_DIR = (HERE / '..').resolve()
_PYPY_INTERPRETER = _REPO_DIR / '.venv-pypy' / 'bin' / 'python'
_TMP_DIR = DATA_DIR / 'tmp'

_CallableT = t.TypeVar('_CallableT', bound=t.Callable)

def _get_module(func: t.Callable) -> str:
    if func.__module__ != '__main__': return func.__module__
    else: return (
            __file__
            .split(str(_REPO_DIR))[1][1:]
            .replace('/', '.')
            .split('.py')[0]
    )


def run_in_pypy(
        func: _CallableT,
        interpreter = _PYPY_INTERPRETER,
) -> _CallableT:
    _TMP_DIR.mkdir(exist_ok=True, parents=True)

    def wrapper(*args, **kwargs):
        _LOG.info(f'Calling in PyPy: {func.__name__} from {_get_module(func)}')
        with (
                NamedTemporaryFile(
                    'w', encoding='utf-8', dir=_TMP_DIR,
                    suffix='.py'
                ) as f_code,
                NamedTemporaryFile('wb', dir=_TMP_DIR) as f_return,
                NamedTemporaryFile('wb', dir=_TMP_DIR) as f_params,
        ):
            f_params.write(json.dumps({
                'args': args,
                'kwargs': kwargs,
            }).encode('utf-8'))
            f_params.flush()

            f_code.write(f'''\
import importlib
import json
import logging
import bgpsyche.logging_config

_LOG = logging.getLogger(__name__)
# _LOG.info('hi')
importlib.import_module('{_get_module(func)}')

with open('{f_params.name}', 'rb') as f:
    params = json.loads(f.read().decode('utf-8'))

ret = {_get_module(func)}.{func.__name__}(*params['args'], **params['kwargs'])
with open('{f_return.name}', 'wb') as f:
    f.write(json.dumps(ret).encode('utf-8'))
            ''')
            f_code.flush()

            module_name = Path(f_code.name).name.split('.py')[0]
            subprocess.run(
                [
                    interpreter, '-m',
                    f'bgpsyche.data.tmp.{module_name}'
                ],
                cwd=_REPO_DIR,
                # capture_output=True,
                encoding='utf-8',
                stdout=stdout,
            )

            with open(f_return.name, 'r', encoding='utf-8') as f:
                ret = json.loads(f.read())

            return ret
            
    return t.cast(_CallableT, wrapper)


def _hello(name: str) -> str: return f'Hello, {name}!'

if __name__ == '__main__':
    print(run_in_pypy(_hello)('World'))