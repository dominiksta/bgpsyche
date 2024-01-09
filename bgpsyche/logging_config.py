import traceback
import sys
import logging
from platform import python_implementation
from datetime import datetime
import os
from pathlib import Path

_LOG_FORMAT = (
    '%(asctime)s %(levelname)-3s [%(module)s] ' +
    ('' if python_implementation() == 'CPython' else '[PyPy] ') +
    '%(message)s'
)
_LOG_FORMAT_DATE = '%y-%m-%d %H:%M:%S'

# file output
# ----------------------------------------------------------------------

_now = datetime.now()
_logging_file_name = Path(os.path.dirname(__file__)) \
    / 'data' / 'logs' / f'{_now.year}_{_now.month}.log'

_logging_file_name.parent.mkdir(exist_ok=True, parents=True)

_file_handler = logging.FileHandler(_logging_file_name)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_FORMAT_DATE))

# stdout
# ----------------------------------------------------------------------

class ColorFormatter(logging.Formatter):

    _BRIGHT_CYAN = "\x1b[37m"
    _YELLOW      = "\x1b[33m"
    _RED         = "\x1b[31m"
    _BOLD_RED    = "\x1b[31;1m"
    _RESET       = "\x1b[0m"

    _FORMATS = {
         logging.DEBUG    : _BRIGHT_CYAN + _LOG_FORMAT + _RESET,
         logging.INFO     :                _LOG_FORMAT         ,
         logging.WARNING  : _YELLOW      + _LOG_FORMAT + _RESET,
         logging.ERROR    : _RED         + _LOG_FORMAT + _RESET,
         logging.CRITICAL : _BOLD_RED    + _LOG_FORMAT + _RESET,
    }

    def format(self, record):
        log_fmt = self._FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, _LOG_FORMAT_DATE)
        return formatter.format(record)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(ColorFormatter())

# general
# ----------------------------------------------------------------------

logging.addLevelName(logging.INFO     , 'INF')
logging.addLevelName(logging.DEBUG    , 'DBG')
logging.addLevelName(logging.WARNING  , 'WRN')
logging.addLevelName(logging.ERROR    , 'ERR')
logging.addLevelName(logging.CRITICAL , 'CRT')

logging.basicConfig(
    level=logging.INFO,
    datefmt=_LOG_FORMAT_DATE,
    handlers=[ _file_handler, _stream_handler ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# logging exceptions on exit
# ----------------------------------------------------------------------

def log_fatal(_type, _value, _traceback):
    get_logger(__name__).critical(
        ''.join(traceback.format_exception(_type, _value, _traceback))
    )

sys.excepthook = log_fatal
