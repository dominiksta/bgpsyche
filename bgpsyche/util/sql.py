from functools import wraps
from traceback import format_exception
import typing as t
from contextlib import closing
import sqlite3
import logging
from pathlib import Path
from time import sleep

_LOG = logging.getLogger(__name__)

def sqlite3_connect_retry(file: Path, timeout_s=10) -> closing[sqlite3.Connection]:
    try:
        return closing(sqlite3.connect(file, timeout=timeout_s))
    except sqlite3.Error as e:
        _LOG.warning(
            f'Could not connect to DB {file.name} after {timeout_s}s, ' +
            f'retrying. Error: {e}'
        )
        sleep(5)
        return sqlite3_connect_retry(file)
