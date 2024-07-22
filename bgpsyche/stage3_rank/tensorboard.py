import typing as t
from datetime import datetime
import logging

from torch.utils.tensorboard import SummaryWriter # type: ignore
from bgpsyche.util.const import DATA_DIR

_LOG = logging.getLogger(__name__)

_TENSORBOARD_DIR = DATA_DIR / 'tensorboard'

_WRITER: t.Optional[SummaryWriter] = None

def make_tensorboard_writer(name: str):
    global _WRITER
    _LOG.info(f'Creating tensorboard writer named: {name}')
    _WRITER = SummaryWriter(
        _TENSORBOARD_DIR /
        f'{datetime.now().strftime("%m.%d_%H.%M.%S")}_{name}'
    )

def tsw() -> SummaryWriter:
    global _WRITER
    assert _WRITER is not None
    return _WRITER