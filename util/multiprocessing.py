import typing as t
import sys
import traceback

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


