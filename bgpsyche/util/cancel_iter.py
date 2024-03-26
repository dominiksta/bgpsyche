import logging
import signal
from types import FrameType
import typing as t

_LOG = logging.getLogger(__name__)

_T = t.TypeVar('_T')

def cancel_iter(
        iter: t.Iterable[_T],
        name: str = '',
):
    cancel = False
    if name == '': name = str(iter)

    sigint_orig_handler = signal.getsignal(signal.SIGINT)
    def sigint_handler(sig: int, frame: t.Optional[FrameType]):
        nonlocal cancel, sigint_orig_handler
        _LOG.warning(f"Iter '{name}' cancelled because SIGINT (Ctrl+C)")
        signal.signal(signal.SIGINT, sigint_orig_handler)
        cancel = True
    signal.signal(signal.SIGINT, sigint_handler)

    for el in iter:
        if cancel: break
        yield el

    signal.signal(signal.SIGINT, sigint_orig_handler)