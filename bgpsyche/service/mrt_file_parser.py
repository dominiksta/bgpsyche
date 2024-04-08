from datetime import datetime
from ipaddress import ip_network
import multiprocessing
import os
import sqlite3
import typing as t
from pathlib import Path
import logging
from hashlib import sha256

import pybgpkit_parser as bgpkit # type: ignore
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.bgp.path_prepending \
    import eliminate_path_prepending as _eliminate_path_prepending
from bgpsyche.util.const import HERE
from bgpsyche.util.ds import str_to_set
from bgpsyche.util.net.typ import IPNetwork
from bgpsyche.util.retry import retry

_LOG = logging.getLogger(__name__)

_TABLE_META = 'mrt_raw_meta'
_TABLE_DATA = 'mrt_raw_data'

_CACHE_DIR = HERE / 'data' / 'mrt_cache'

class BgpElem(t.TypedDict):
    """from bgpkit"""
    timestamp: int
    elem_type: t.Literal['A', 'W'] # announce/withdraw
    peer_ip: str
    peer_asn: int
    prefix: str
    next_hop: t.Optional[str]
    as_path: t.Optional[str]
    origin_asns: t.Optional[t.List[int]]
    origin: t.Optional[t.Literal['IGP' 'EGP', 'INCOMPLETE']]
    local_pref: t.Optional[int]
    med: t.Optional[int]
    communities: t.Optional[t.List[str]]
    atomic: t.Optional[t.Literal['NAG', 'AG']]
    aggr_asn: t.Optional[int]
    aggr_ip: t.Optional[str]
    only_to_customer: t.Optional[int]


def iter_mrt_file_raw(
        file: Path,
) -> t.Iterator[BgpElem]:
    parser = t.cast(t.Iterable[BgpElem], bgpkit.Parser(str(file))) # type: ignore
    _LOG.info(f'Now parsing {file.name}')
    iter = 0
    for elem in parser:
        iter += 1
        if iter % 100_000 == 0: _LOG.info(f'{file.name}: Parsed {iter} routes')
        assert elem['elem_type'] == 'A'
        yield elem


def iter_mrt_file_paths(
        file: Path,
        skip_with_cycle = True,
        skip_length_one = True,
        assert_only_announcements = True,
) -> t.Iterator[t.Tuple[t.List[int], IPNetwork]]:
    for elem in iter_mrt_file_raw(file):
        assert (not assert_only_announcements) or elem['elem_type'] == 'A'

        path = elem['as_path'].split(' ')
        if skip_length_one and len(path) == 1: continue

        path_int: t.List[int] = []
        for i in range(0, len(path)):
            # TODO: what even is this?
            if '{' in path[i]: path_int.append(int(list(str_to_set(path[i]))[0]))
            else: path_int.append(int(path[i]))

        if skip_with_cycle and \
           False in [
               _eliminate_path_prepending(path_int).count(asn) == 1
               for asn in path_int
           ]:
            # print(f'Cycle: {path_int}')
            continue

        yield path_int, ip_network(elem['prefix'])


def _prepare_sqlite_single(args):
    mrt_file: Path               = args[0]
    sqlite_file: Path            = args[1]

    BUFFER_SIZE = 100_000

    @retry()
    def _insert(buf: t.List[t.Tuple[int, int, str, str]]):
        with sqlite3.connect(sqlite_file, timeout=120) as tx:
            tx.executemany(f"""
                INSERT INTO {_TABLE_DATA}
                (as_source, as_sink, as_path, prefix)
                VALUES (?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET count=excluded.count+1
            """, buf)

    buffer: t.List[t.Tuple[int, int, str, str]] = []
    for path, prefix in iter_mrt_file_paths(
            mrt_file,
    ):
        path_str = ' '.join(map(str, path))
        buffer.append((path[0], path[-1], path_str, str(prefix)))
        if len(buffer) % BUFFER_SIZE == 0:
            # _LOG.info(f'Inserting {BUFFER_SIZE} paths into table {table_name}')
            _insert(buffer)
            buffer = []

    _insert(buffer)


def _prepare_sqlite(
        sqlite_file: Path,
        mrt_files: t.List[Path],
        workers: int = -1,
):
    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        tx.executescript(f"""
          DROP TABLE IF EXISTS {_TABLE_DATA};

          CREATE TABLE {_TABLE_DATA} (
                as_source   INTEGER NOT NULL,
                as_sink     INTEGER NOT NULL,
                as_path     TEXT NOT NULL,
                prefix      TEXT NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(as_path, prefix)
          );

          -- CREATE INDEX IF NOT EXISTS i_{_TABLE_DATA}_source
          --   ON {_TABLE_DATA} (as_source);
          -- CREATE INDEX IF NOT EXISTS i_{_TABLE_DATA}_sink
          --   ON {_TABLE_DATA} (as_sink);

          DELETE FROM {_TABLE_DATA};

          DELETE FROM {_TABLE_META};
        """)
        tx.execute(
            f'INSERT INTO {_TABLE_META} VALUES (?)',
            ('\n'.join(map(str, mrt_files)),)
        )


    progress = Progress(len(mrt_files), f'Parsing MRT files {sqlite_file.name}')

    args = [ (f, sqlite_file) for f in mrt_files ]

    if workers == -1: workers = (os.cpu_count() or 3) - 2
    with multiprocessing.Pool(workers) as p:
        ret = []
        for ret_single in p.imap_unordered(
            _prepare_sqlite_single, args, chunksize=1
        ):
            ret.append(ret_single)
            progress.update()

    progress.complete()


def _check_sqlite(
        sqlite_file: Path,
        mrt_files: t.List[Path],
) -> bool:
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        tx.executescript(f"""
          CREATE TABLE IF NOT EXISTS {_TABLE_META} (
                paths   TEXT NOT NULL,
                PRIMARY KEY(paths)
          );
        """)

        paths_resp = tx.execute(f'SELECT * FROM {_TABLE_META}').fetchone()

    if paths_resp is None: return False
    paths_resp = paths_resp[0]

    paths_in_db = [ Path(p) for p in paths_resp.split('\n') ]

    for path in paths_in_db:
        if path not in mrt_files: return False

    for path in mrt_files:
        if path not in paths_in_db: return False

    return True


class ASPathMeta(t.TypedDict):
    count: int
    path: t.List[int]

class ASPathMetaWithPrefix(t.TypedDict):
    count: int
    path: t.List[int]
    dst_prefix: IPNetwork

@t.overload
def iter_paths(
        mrt_files: t.List[Path],
        distinct_paths: t.Literal[True] = True,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = True,
        sqlite_cache_file_prefix = '',
) -> t.Iterator[ASPathMeta]: pass

@t.overload
def iter_paths(
        mrt_files: t.List[Path],
        distinct_paths: t.Literal[False],
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = True,
        sqlite_cache_file_prefix = '',
) -> t.Iterator[ASPathMetaWithPrefix]: pass

def iter_paths(
        mrt_files: t.List[Path],
        distinct_paths: bool = True,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = True,
        sqlite_cache_file_prefix = '',
) -> t.Union[
    t.Iterator[ASPathMeta],
    t.Iterator[ASPathMetaWithPrefix],
]:
    sqlite_id = sha256(
        ';;'.join([ str(p.absolute()) for p in mrt_files ]).encode('UTF-8')
    ).hexdigest()

    sqlite_date_prefix = datetime.now().strftime('%Y%m')

    sqlite_file = _CACHE_DIR \
        / f'{sqlite_cache_file_prefix}_{sqlite_date_prefix}_{sqlite_id}.sqlite3'

    _LOG.info(f'Preparting to loop over paths: {sqlite_file.name}')

    if not _check_sqlite(sqlite_file, mrt_files):
        _LOG.info(f'DB {sqlite_file.name} "cache miss" for MRT files')
        _prepare_sqlite(sqlite_file, mrt_files)

    query = {
        True  :
            f'SELECT as_path, count ' +
            f' FROM {_TABLE_DATA} GROUP BY as_path',
        False :
            f'SELECT as_path, count, prefix ' +
            f' FROM {_TABLE_DATA}',
    }[distinct_paths]

    _LOG.info(f'Running Query: {query}')

    filter_sources = t.cast(t.Set[int], filter_sources) # shut up mypy
    filter_sinks = t.cast(t.Set[int], filter_sinks)

    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        # not exact for the grouping query, but good enough and much faster
        tbl_size = tx.execute(
            f'SELECT COUNT({"DISTINCT as_path" if distinct_paths else "*"}) ' +
            f'FROM {_TABLE_DATA}'
        ).fetchone()[0]
        # assert tbl_size >= 10_000

        iter = 0
        for row in tx.execute(query):
            path_str: str           = row[0]
            if eliminate_path_prepending:
                path_str = _eliminate_path_prepending(path_str)
            path_int: t.List[int]   = list(map(int, path_str.split(' ')))
            count: int              = row[1]

            iter += 1
            if iter % 100_000 == 0:
                _LOG.info(f'Got {iter}/{tbl_size} rows from {_TABLE_DATA}')

            if len(path_int) == 1: continue

            # while we could do this in sql, that would put a hard limit on the
            # amount of sources/sinks we can filter because there can only be so
            # many elemnts in an IN clause or so many AND clauses in a WHERE
            if filter_sources is not None \
               and path_int[0] not in filter_sources: continue
            if filter_sinks is not None \
               and path_int[-1] not in filter_sinks: continue

            if distinct_paths:
                yield {
                    'path': path_int,
                    'count': count,
                }
            else:
                yield {
                    'path': path_int,
                    'count': count,
                    'dst_prefix': ip_network(row[2]),
                }
