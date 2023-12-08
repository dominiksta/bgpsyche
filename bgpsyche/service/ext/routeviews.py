import itertools
import typing as t
from datetime import datetime
from pathlib import Path

from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.net.download import download_file_cached
from bgpsyche.service import mrt_file_parser

_BASE_URL = 'https://archive.routeviews.org'
_DATA_DIR = DATA_DIR / 'mrt_routeviews'

# collector list from https://archive.routeviews.org/

COLLECTORS_BASE = [
    'route-views2', 'route-views3', 'route-views4',
    'route-views5', 'route-views6',
]

COLLECTORS_IXP = [
    'decix.jhb', 'route-views.amsix', 'route-views.chicago',
    'route-views.chile', 'route-views.eqix', 'route-views.flix',
    'route-views.gorex', 'route-views.isc', 'route-views.kixp',
    'route-views.jinx', 'route-views.linx', 'route-views.napafrica',
    'route-views.nwax', 'pacwave.lax', 'pit.scl',
    'route-views.phoix', 'route-views.telxatl', 'route-views.wide',
    'route-views.sydney', 'route-views.saopaulo', 'route-views2.saopaulo',
    'route-views.sg', 'route-views.perth', 'route-views.peru',
    'route-views.sfmix', 'route-views.siex', 'route-views.soxrs',
    'route-views.mwix', 'route-views.rio', 'route-views.fortaleza',
    'route-views.gixa', 'route-views.bdix', 'route-views.bknix',
    'route-views.uaeix', 'route-views.ny',
]


def download_single_full_table(
        date: datetime, out_dir: Path, collector: str
) -> Path:
    collector_url = f'/{collector}' if collector != 'route-views2' else ''
    # HACK: currently only works with "exact" datetimes matching the filenames
    filename = date.strftime('rib.%Y%m%d.%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    return download_file_cached(
        _BASE_URL + collector_url +
        f'/bgpdata/{date.strftime("%Y.%m")}/RIBS/{filename}.bz2',
        out_dir / (filename + f'.{collector}.bz2')
    )


def download_all_full_tables(
        date: datetime,
        out_dir: Path = _DATA_DIR,
        collectors: t.List[str] = COLLECTORS_BASE,
) -> t.List[Path]:
    return [
        download_single_full_table(date, out_dir, col)
        for col in collectors
    ]


def iter_raw(
        dt: datetime,
        collectors: t.List[str] = COLLECTORS_BASE,
) -> t.Iterator[mrt_file_parser.BgpElem]:
    return itertools.chain(*[
        mrt_file_parser.iter_mrt_file_raw(path)
        for path in download_all_full_tables(dt, _DATA_DIR, collectors)
    ])


def iter_paths(
        dt: datetime,
        collectors: t.List[str] = COLLECTORS_BASE,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Iterator[mrt_file_parser.ASPathMeta]:
    return mrt_file_parser.iter_paths(
        mrt_files=download_all_full_tables(dt, _DATA_DIR, collectors),
        include_origin=False,
        filter_sinks=filter_sinks,
        filter_sources=filter_sources,
        eliminate_path_prepending=eliminate_path_prepending,
        sqlite_cache_file_prefix='routeviews',
    )