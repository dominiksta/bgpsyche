import os.path
import typing as t
from pathlib import Path
import time
import logging
import urllib.request

_LOG = logging.getLogger(__name__)

def download_file_cached(url: str, filename: t.Union[Path, str]) -> Path:
    last_report_print = time.time()
    f = Path(filename)
    f.parent.mkdir(parents=True, exist_ok=True)

    def report_hook(block_num: int, block_size: int, total_size: int):
        nonlocal last_report_print
        downloaded = round(block_num * block_size / 2**20, 2)
        now = time.time()
        if now - last_report_print > 1:
            _LOG.info(
                f'{f.name}: {downloaded}MB /' +
                f' {round(total_size / 2 ** 20, 2)}MB'
            )
            last_report_print = now

    if not os.path.exists(f):
        _LOG.info(f'Downloading {url}')
        urllib.request.urlretrieve(url, f, report_hook)

    return Path(filename)
