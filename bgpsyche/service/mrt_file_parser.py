from ipaddress import ip_network
import multiprocessing
import os
import sqlite3
import typing as t
from pathlib import Path
import logging

import pybgpkit_parser as bgpkit
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.bgp.path_prepending \
    import eliminate_path_prepending as _eliminate_path_prepending
from bgpsyche.util.ds import str_to_set
from bgpsyche.util.net.typ import IPNetwork

_LOG = logging.getLogger(__name__)

_TABLE_META = 'mrt_raw_meta'
_TABLE_DATA = 'mrt_raw_data'

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
    parser = t.cast(t.Iterable[BgpElem], bgpkit.Parser(str(file)))
    _LOG.info(f'Now parsing {file.name}')
    iter = 0
    for elem in parser:
        iter += 1
        if iter % 100000 == 0: _LOG.info(f'{file.name}: Parsed {iter} routes')
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
           False in [ path_int.count(asn) == 1 for asn in path_int ]:
            continue

        yield path_int, ip_network(elem['prefix'])


def _prepare_sqlite_single(args):
    mrt_file: Path               = args[0]
    sqlite_file: Path            = args[1]
    origin_name: t.Optional[str] = args[2]

    BUFFER_SIZE = 100_000
    _origin_name = origin_name if origin_name is not None else mrt_file.name

    buffer: t.List[t.Tuple[int, int, str, str, str]] = []
    for path, prefix in iter_mrt_file_paths(
            mrt_file,
    ):
        path_str = ' '.join(map(str, path))
        buffer.append((path[0], path[-1], path_str, _origin_name, str(prefix)))
        if len(buffer) % BUFFER_SIZE == 0:
            # _LOG.info(f'Inserting {BUFFER_SIZE} paths into table {table_name}')
            with sqlite3.connect(sqlite_file, timeout=120) as tx:
                tx.executemany(f"""
                  INSERT INTO {_TABLE_DATA}
                    (as_source, as_sink, as_path, origin, prefix)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET count=excluded.count+1
                """, buffer)
            buffer = []

    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        tx.executemany(f"""
          INSERT INTO {_TABLE_DATA}
            (as_source, as_sink, as_path, origin, prefix)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET count=excluded.count+1
        """, buffer)


def _prepare_sqlite(
        sqlite_file: Path,
        mrt_files: t.List[Path],
        origin_name: t.Optional[str] = None,
        workers: int = -1,
):
    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        tx.executescript(f"""
          DROP TABLE IF EXISTS {_TABLE_DATA};

          CREATE TABLE {_TABLE_DATA} (
                as_source   INTEGER NOT NULL,
                as_sink     INTEGER NOT NULL,
                as_path     TEXT NOT NULL,
                origin      TEXT NOT NULL,
                prefix      TEXT NOT NULL,
                count       INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(as_path, origin, prefix)
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

    args = [
        (f, sqlite_file, origin_name) for f in mrt_files
    ]

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
    dst_prefix: IPNetwork

class ASPathMetaWithOrigin(ASPathMeta):
    origin: str

@t.overload
def iter_paths(
        sqlite_file: Path,
        mrt_files: t.List[Path],
        include_origin: t.Literal[False],
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Iterator[ASPathMeta]: pass

@t.overload
def iter_paths(
        sqlite_file: Path,
        mrt_files: t.List[Path],
        include_origin: t.Literal[True],
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Iterator[ASPathMetaWithOrigin]: pass

def iter_paths(
        sqlite_file: Path,
        mrt_files: t.List[Path],
        include_origin: bool = False,
        filter_sources: t.Optional[t.Set[int]] = None,
        filter_sinks: t.Optional[t.Set[int]] = None,
        eliminate_path_prepending: bool = False,
) -> t.Union[
    t.Iterator[ASPathMetaWithOrigin],
    t.Iterator[ASPathMeta],
]:
    if not _check_sqlite(sqlite_file, mrt_files):
        _LOG.info(f'DB {sqlite_file.name} "cache miss" for MRT files')
        _prepare_sqlite(sqlite_file, mrt_files)

    query = {
        True  : f'SELECT as_path, count, prefix, origin FROM {_TABLE_DATA}',
        False :
            f'SELECT as_path, SUM(count), prefix ' +
            f' FROM {_TABLE_DATA} GROUP BY as_path',
    }[include_origin]

    filter_sources = t.cast(t.Set[int], filter_sources) # shut up mypy
    filter_sinks = t.cast(t.Set[int], filter_sinks)

    with sqlite3.connect(sqlite_file, timeout=120) as tx:
        # not exact for the grouping query, but good enough and much faster
        tbl_size = tx.execute(
            f'SELECT COUNT(*) FROM {_TABLE_DATA}'
        ).fetchone()[0]
        assert tbl_size >= 100_000

        iter = 0
        for row in tx.execute(query):
            path_str: str           = row[0]
            if eliminate_path_prepending:
                path_str = _eliminate_path_prepending(path_str)
            path_int: t.List[int]   = list(map(int, path_str.split(' ')))
            count: int              = row[1]
            prefix: IPNetwork       = ip_network(row[2])

            iter += 1
            if iter % 100_000 == 0:
                _LOG.info(f'Got {iter}/{tbl_size} rows from {_TABLE_DATA}')

            # while we could do this in sql, that would put a hard limit on the
            # amount of sources/sinks we can filter because there can only be so
            # many elemnts in an IN clause or so many AND clauses in a WHERE
            if filter_sources is not None \
               and path_int[0] not in filter_sources: continue
            if filter_sinks is not None \
               and path_int[-1] not in filter_sinks: continue

            if include_origin:
                origin: str = row[3]
                yield {
                    'path': path_int,
                    'count': count,
                    'dst_prefix': prefix,
                    'origin': origin,
                }
            else:
                yield {
                    'path': path_int,
                    'count': count,
                    'dst_prefix': prefix,
                }
