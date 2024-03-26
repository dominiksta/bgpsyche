from datetime import datetime

from torch.utils.tensorboard import SummaryWriter # type: ignore
from bgpsyche.util.const import DATA_DIR

_tensorboard_dir = DATA_DIR / 'tensorboard'
tensorboard_writer = SummaryWriter(
    _tensorboard_dir /
    f'{datetime.now().strftime("%m.%d_%H.%M.%S")}_{input("Name run: ")}'
)

