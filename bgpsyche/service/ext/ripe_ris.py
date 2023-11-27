import itertools
import logging
import typing as t
from datetime import datetime
from pathlib import Path

from bgpsyche.service import mrt_file_parser
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net import download_file_cached

_RIS_BASE_URL = 'https://data.ris.ripe.net/'
_DATA_DIR = DATA_DIR / 'mrt_ris'

_LOG = logging.getLogger(__name__)

def download_single_full_table(
        date: datetime, out_dir: Path, collector: str
) -> Path:
    # HACK: currently only works with "exact" datetimes matching the filenames
    filename = date.strftime('bview.%Y%m%d.%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    return download_file_cached(
        _RIS_BASE_URL +
        f'{collector}/{date.strftime("%Y.%m")}/{filename}.gz',
        out_dir / (filename + f'.{collector}.gz')
    )


# https://ris.ripe.net/docs/10_routecollectors.html#bgp-timer-settings
ACTIVE_COLLECTORS = [
    'rrc00', 'rrc01', 'rrc03', 'rrc04', 'rrc05', 'rrc06', 'rrc07', 'rrc10',
    'rrc11', 'rrc12', 'rrc13', 'rrc14', 'rrc15', 'rrc16', 'rrc18', 'rrc19',
    'rrc20', 'rrc21', 'rrc22', 'rrc23', 'rrc24', 'rrc25', 'rrc26',
]


def download_all_full_tables(
        date: datetime, out_dir: Path,
        collectors: t.List[str] = ACTIVE_COLLECTORS,
) -> t.List[Path]:
    return [
        download_single_full_table(date, out_dir, col)
        for col in collectors
    ]


def iter_raw(dt: datetime) -> t.Iterator[mrt_file_parser.BgpElem]:
    return itertools.chain(*[
        mrt_file_parser.iter_mrt_file_raw(path)
        for path in download_all_full_tables(dt, _DATA_DIR)
    ])


def iter_paths(
        dt: datetime,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Iterator[mrt_file_parser.ASPathMeta]:
    _LOG.info('Starting to loop over RIPE RIS paths')
    return mrt_file_parser.iter_paths(
        sqlite_file=_DATA_DIR / 'db' / f'{dt.strftime("%Y-%m-%dT%H%M")}.sqlite3',
        mrt_files=download_all_full_tables(dt, _DATA_DIR),
        include_origin=False,
        filter_sinks=filter_sinks,
        filter_sources=filter_sources,
        eliminate_path_prepending=eliminate_path_prepending,
    )