import logging
from os import cpu_count
import typing as t
import sys
import traceback
import math

from bgpsyche.util.os import get_memory

_LOG = logging.getLogger(__name__)

def wrap_raise_with_trace(func: t.Callable, *args, **kwargs):
    """
    Given a function, run it and return its result. We can use this with
    multiprocessing.map and map it over a list of job functors to do them.

    Handles getting more than multiprocessing's pitiful exception output

    Credits: https://stackoverflow.com/a/16618842
    """

    try:
        return func(*args, **kwargs)
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def worker_amount(
        ram_per_worker_mb: t.Optional[int] = None,
        leave_free_cores = 1,
) -> int:
    mem = get_memory()
    cores = cpu_count()
    if cores is None:
        _LOG.warning('Could not determine core count, using default 3')
        cores = 3

    max_workers = math.inf

    if ram_per_worker_mb is not None:
        max_workers = min(int(mem['total_mb'] / ram_per_worker_mb), max_workers)

    max_workers = min(cores - leave_free_cores, max_workers)
    return int(max_workers)