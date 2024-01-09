import functools
import inspect
import json
import logging
from pathlib import Path
from platform import python_implementation
from pprint import pformat
from sys import stdout
import typing as t
import subprocess
from tempfile import NamedTemporaryFile
from bgpsyche.caching.base import SerializableFileCache

import bgpsyche.logging_config
from bgpsyche.util.const import DATA_DIR, HERE
from bgpsyche.util.platform import get_func_module

"""
Here be dragons.
"""

_LOG = logging.getLogger(__name__)

_REPO_DIR = (HERE / '..').resolve()
_PYPY_INTERPRETER = _REPO_DIR / '.venv-pypy' / 'bin' / 'python'
_TMP_DIR = DATA_DIR / 'tmp'

_CallableT = t.TypeVar('_CallableT', bound=t.Callable)


def run_in_pypy(
        module: t.Optional[str] = None,
        interpreter = _PYPY_INTERPRETER,
        cache: t.Optional[t.Type[SerializableFileCache]] = None,
) -> t.Callable[[_CallableT], _CallableT]:
    _TMP_DIR.mkdir(exist_ok=True, parents=True)
    def run_in_pypy_wrapper(func: _CallableT) -> _CallableT:
        if python_implementation() != 'CPython': return func
        nonlocal module
        if module is None: module = get_func_module(func)

        @functools.wraps(func)
        def run_in_pypy_inner(*args, **kwargs):
            try:
                json.dumps(args); json.dumps(kwargs)
            except:
                _LOG.critical(
                    f'Could not serialize arguments as JSON: ' +
                    f'{pformat({"args": args, "kwargs": kwargs})}'
                )
                raise


            if cache is not None:
                _cache = cache(f'{module}.{func.__name__}', lambda: None)
                params: t.Any = _cache._get_meta()['custom'] or { 'args': [], 'kwargs': {} }
                if params['args'] != list(args):
                    _cache.invalidate()
                    _LOG.info(f'Cache invalidated, {pformat((args, params["args"]))}')
                for key in kwargs.keys():
                    if params['kwargs'][key] != kwargs[key]:
                        _cache.invalidate()
                        _LOG.info(f'Cache invalidated, {pformat((args, params["args"]))}')

                _cache._set_meta(custom={ 'args': args, 'kwargs': kwargs })

                if _cache.check(): return _cache.get()

            _LOG.info(f'Calling in PyPy: {func.__name__} from {module}')
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

                code = f'''\
import importlib
import json
import logging
import bgpsyche.logging_config
import bgpsyche.caching                

_LOG = logging.getLogger(__name__)
# _LOG.info('hi')
importlib.import_module('{module}')

with open('{f_params.name}', 'rb') as f:
    params = json.loads(f.read().decode('utf-8'))
    
if {cache is None}:
    ret = {module}.{func.__name__}(*params['args'], **params['kwargs'])
    with open('{f_return.name}', 'wb') as f:
        f.write(json.dumps(ret).encode('utf-8'))
else:
    bgpsyche.caching.{cache.__name__}(
        '{module}.{func.__name__}',
        lambda: {module}.{func.__name__}(*params['args'], **params['kwargs'])
    ).get()
                '''
                # print(code)
                f_code.write(code)
                f_code.flush()

                module_name = Path(f_code.name).name.split('.py')[0]
                completed = subprocess.run(
                    [
                        interpreter, '-m',
                        f'bgpsyche.data.tmp.{module_name}'
                    ],
                    cwd=_REPO_DIR,
                    # capture_output=True,
                    encoding='utf-8',
                    stdout=stdout,
                )
                if completed.returncode != 0:
                    raise ValueError(f'Non-zero return code {completed.returncode}')

                if cache is None:
                    with open(f_return.name, 'r', encoding='utf-8') as f:
                        ret = json.loads(f.read())
                else:
                    ret = _cache.get() # type: ignore

                return ret
            
        return t.cast(_CallableT, run_in_pypy_inner)
    return t.cast(t.Any, run_in_pypy_wrapper)


def _hello(name: str) -> str: return f'Hello, {name}!'

if __name__ == '__main__':
    print(run_in_pypy()(_hello)('World'))