import contextlib
from pathlib import Path
from time import sleep
import typing as t
import sqlite3
import logging
from bgpsyche.stage1_candidates.from_graph import GetPathCandidatesAbortConditions, abort_on_amount, abort_on_timeout

from bgpsyche.util.const import DATA_DIR
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates

_LOG = logging.getLogger(__name__)

def _connect(file: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(file, timeout=10)
    except sqlite3.Error as e:
        _LOG.warning(
            f'Could not connect to DB after 10s, retrying. Error: {e}'
        )
        sleep(5)
        return _connect(file)

class PathCandidateCache:

    def __init__(
            self, name: str,
            abort_customer_cone_search: GetPathCandidatesAbortConditions = lambda: [
                { 'func': abort_on_timeout(1), 'desc': 'timeout 1s' },
                { 'func': abort_on_amount(4000), 'desc': 'amount 4k' },
            ],
            abort_full_search: GetPathCandidatesAbortConditions = lambda: [
                { 'func': abort_on_timeout(5), 'desc': 'timeout 5s' },
                { 'func': abort_on_amount(4000), 'desc': 'amount 4k' },
            ],
            quiet=False,
    ) -> None:
        self.name = name
        self._quiet = quiet
        self._abort_customer_cone_search = abort_customer_cone_search
        self._abort_full_search = abort_full_search
        self._cache_db_path = \
            DATA_DIR / 'cache' / 'path_candidates' / f'{name}.sqlite3'
        self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = lambda: contextlib.closing(_connect(self._cache_db_path))

        with self._con() as con, con as tx:
            tx.execute("""
              CREATE TABLE IF NOT EXISTS paths (
                as_source   INTEGER NOT NULL,
                as_sink     INTEGER NOT NULL,
                as_path     TEXT NOT NULL
              )
            """)


    def init_caches(self) -> 'PathCandidateCache':
        get_path_candidates(3320, 3320)
        return self


    def get(self, source: int, sink: int) -> t.List[t.List[int]]:
        candidates: t.List[t.List[int]] = []
        with self._con() as con, con as tx:
            resp = list(tx.execute(
                'SELECT * FROM paths WHERE as_source = ? AND as_sink = ?',
                (source, sink)
            ))
            if len(resp) == 0:
                if not self._quiet: _LOG.info(f'Path cache MISS {source} -> {sink}')
                candidates = list(get_path_candidates(
                    source, sink,
                    abort_customer_cone_search=self._abort_customer_cone_search,
                    abort_full_search=self._abort_full_search,
                    quiet=True
                ))
                tx.executemany(
                    'INSERT INTO paths VALUES (?, ?, ?)',
                    ((source, sink, ' '.join(map(str, path))) for path in candidates)
                )
            else:
                if not self._quiet: _LOG.info(f'Path cache HIT {source} -> {sink}')
                candidates = [
                    [ int(asn) for asn in path.split(' ') ] for _, __, path in resp
                ]
        return candidates


    def invalidate(self) -> None:
        with self._con() as tx: tx.execute('DELETE FROM paths')
