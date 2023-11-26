import functools
import typing as t
from datetime import datetime, timedelta
import logging

_LOG = logging.getLogger(__name__)

_CallableT = t.TypeVar('_CallableT', bound=t.Callable)

# benchmarking (measuring time)
# ----------------------------------------------------------------------

BenchStart = t.Tuple[str, datetime, bool]

def bench_start(name: str, only_in_debug: bool = False) -> BenchStart:
    now = datetime.now()
    if not (only_in_debug and not __debug__):
        _LOG.info(f'bench_start <{name}>')
    return name, now, only_in_debug


def bench_end(start: BenchStart) -> timedelta:
    now = datetime.now()
    if not (start[2] and not __debug__):
        _LOG.info(f'bench_end <{start[0]}>. Took {(now - start[1])}')
    return now - start[1]


def bench_function(func: _CallableT) -> _CallableT:
    @functools.wraps(func)
    def wrapper_bench(*args, **kwargs):
        bench = bench_start('function ' + func.__name__)
        ret = func(*args, **kwargs)
        bench_end(bench)
        return ret
    return t.cast(_CallableT, wrapper_bench)


# Progress (ETA)
# ----------------------------------------------------------------------

class Progress:
    def __init__(
            self, total: int, name: str
    ) -> None:
        self.__bench = bench_start(name)
        self.__count: int = 0
        self.__last_update: t.Optional[datetime] = None
        self.__took: t.List[float] = []
        self.__name = name
        self.__total = total
        self.__took_total: t.Optional[timedelta] = None

    def update(self, msg: t.Optional[str] = None) -> None:
        now = datetime.now()
        self.__count += 1

        eta: t.Optional[timedelta] = None
        if self.__last_update is not None:
            self.__took.append((now - self.__last_update).total_seconds())
            eta = timedelta(
                seconds=(
                    (self.__total - self.__count) *
                    (sum(self.__took) / len(self.__took))
                )
            )
            eta -= timedelta(microseconds=eta.microseconds) # for readability

        self.__last_update = now

        _LOG.info(
            f'Progress <{self.__name}> ' +
            f'[{str(self.__count).zfill(round(self.__total ** (1/10) + 2))}' +
            f'/{self.__total}, ETA {eta}' +
            (f', took {round(self.__took[-1], 1)}s' if len(self.__took) != 0 else '') +
            ']' +
            (f': {msg}' if msg is not None else '')
        )

    def complete(self) -> None:
        self.__took_total = bench_end(self.__bench)

    @property
    def took_total(self):
        if self.__took_total is None: raise RuntimeError(
                'The total time can not be queried until `complete` is called'
        )
        return self.__took_total
