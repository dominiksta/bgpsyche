import typing as t

from bgpsyche.service import mrt_file_parser
from bgpsyche.util.const import DATA_DIR

def iter_paths(
        data_dir: str,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Iterator[mrt_file_parser.ASPathMeta]:
    return mrt_file_parser.iter_paths(
        mrt_files=list((DATA_DIR / data_dir).rglob('*.bz2')),
        include_origin=False,
        filter_sinks=filter_sinks,
        filter_sources=filter_sources,
        eliminate_path_prepending=eliminate_path_prepending,
        sqlite_cache_file_prefix=f'mrt_custom_{data_dir}',
    )
