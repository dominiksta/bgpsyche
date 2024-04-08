import typing as t
import inspect

def get_func_module(func: t.Callable) -> str:
    try:
        if func.__module__ != '__main__': return func.__module__
        else: return (
            str(inspect.getmodule(func).__package__) + '.' +
            str(inspect.getmodulename(inspect.getfile(func)))
        )
    except:
        return '__UNKNOWN_FUNCTION__'

