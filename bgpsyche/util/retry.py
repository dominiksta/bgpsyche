from functools import wraps
from traceback import format_exception
import typing as t
from time import sleep
import logging

from bgpsyche.util.platform import get_func_module

_LOG = logging.getLogger(__name__)

_CallableT = t.TypeVar('_CallableT', bound=t.Callable)

def retry(
        retries: t.Union[t.Literal['inf'], int] = 'inf',
        exceptions: t.Tuple[t.Type[Exception]] = (Exception,),
        sleep_base_s: float = 3,
        sleep_multiplier: float = 2,
        sleep_max_s: float = 120,
) -> t.Callable[[_CallableT], _CallableT]:

    sleep_current = sleep_base_s
    
    def inner_wrapper(fun: _CallableT) -> _CallableT:
        @wraps(fun)
        def inner_exec(*args, **kwargs):
            nonlocal retries, sleep_current

            ex: t.Optional[Exception] = None
            while retries == 'inf' or retries > 0:
                try:
                    return fun(*args, **kwargs)
                except exceptions as e:
                    ex = e
                    if retries != 'inf': retries -= 1
                    sleep_current = min(
                        sleep_max_s, sleep_base_s * sleep_multiplier
                    )
                    _LOG.warning(
                        f'Retrying {get_func_module(fun)}.{fun.__name__} ' +
                        f'[sleeping for {sleep_current}s]: ' +
                        f'Error: {ex.__class__.__name__} - {ex}'
                    )
                    sleep(sleep_current)
                    continue

            raise RuntimeError(
                f'Too many retries ({retries}) of {fun.__name__}'
            ) from ex


        return t.cast(_CallableT, inner_exec)
    return t.cast(t.Any, inner_wrapper)
