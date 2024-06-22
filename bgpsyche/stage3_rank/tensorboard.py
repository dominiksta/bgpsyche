from datetime import datetime
import logging

from torch.utils.tensorboard import SummaryWriter # type: ignore
from bgpsyche.util.const import DATA_DIR

_LOG = logging.getLogger(__name__)

_tensorboard_dir = DATA_DIR / 'tensorboard'
_name = input("Name run: ")
_LOG.info(f'Tensorboard run named: {_name}')
tensorboard_writer = SummaryWriter(
    _tensorboard_dir /
    f'{datetime.now().strftime("%m.%d_%H.%M.%S")}_{_name}'
)
