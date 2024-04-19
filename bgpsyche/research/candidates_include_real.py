import itertools
import random
import multiprocessing
import json
from time import time
import typing as t
from datetime import datetime
import logging
import statistics

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.stage1_candidates.from_customer_cones import get_path_candidates_from_customer_cones
from bgpsyche.stage1_candidates.from_full_graph import get_path_candidates_full_graph
from bgpsyche.stage1_candidates.from_graph import GetPathCandidatesAbortConditions
from bgpsyche.stage1_candidates.get_candidates import abort_on_timeout, get_path_candidates
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import DATA_DIR
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = 10
_WORKER_CHUNKSIZE = 10
_RESULT_DIR = DATA_DIR / 'research' / 'results'
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# set get candidate function
# ----------------------------------------------------------------------

_TIMEOUT = 10

_ABORT_COND: t.Callable[
    [str, t.List[int], t.Optional[int]],
    GetPathCandidatesAbortConditions
] = \
    lambda n, p, t: lambda: [
        { 'func': lambda _p: _p == p, 'desc': f'found [{n}]' },
        { 'func': abort_on_timeout(t or _TIMEOUT), 'desc': f'timeout {_TIMEOUT}s [{n}]' }
    ]

_GET_CANDIDATES_ALL = lambda path: get_path_candidates(
    path[0], path[-1],
    abort_customer_cone_search=_ABORT_COND('cone', path, None),
    abort_full_search=_ABORT_COND('full', path, None),
    quiet=True,
)

_GET_CANDIDATES_ALL_PROD = lambda path: get_path_candidates(
    path[0], path[-1],
    abort_customer_cone_search=_ABORT_COND('cone', path, 1),
    abort_full_search=_ABORT_COND('full', path, 3),
    quiet=True,
)

_GET_CANDIDATES_ONLY_FULL = lambda path: get_path_candidates_full_graph(
    path[0], path[-1], _ABORT_COND('full', path, None), quiet=True,
)['candidates']

_GET_CANDIDATES_ONLY_CONE = lambda path: get_path_candidates_from_customer_cones(
    path[0], path[-1], _ABORT_COND('cone', path, None), quiet=True,
)['candidates']

_GET_CANDIDATES = _GET_CANDIDATES_ALL_PROD

# ----------------------------------------------------------------------

def _research_candidates_include_real_worker(args) -> t.Tuple[
        bool, t.Optional[int], t.List[int], float
]:
    path: t.List[int] = args[0]
    before = time()

    candidates = _GET_CANDIDATES(path)

    if path in candidates:
        return True, candidates.index(path), path, round(time() - before, 2)
    else:
        # _LOG.info(f'Path not found: {path}')
        return False, None, path, 0.0


@run_in_pypy()
def _research_candidates_include_real():
    ris_paths = _load_ris_paths()[:10_000]

    # HACK: initialize cache before workers all start populating cache
    get_path_candidates(3320, 3320)

    worker_params = ( (path,) for path in ris_paths )

    iter, included, included_pos, not_included_len, took = 0, 0, [3], [5.0], [2.0]

    prg = Progress(
        round(len(ris_paths) / _WORKER_CHUNKSIZE),
        'total'
    )

    now = datetime.now()

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        for res in p.imap_unordered(
                _research_candidates_include_real_worker,
                worker_params, chunksize=_WORKER_CHUNKSIZE
        ):
            w_included, w_included_pos, w_path, w_took = t.cast(
                t.Tuple[bool, t.Optional[int], t.List[int], float], res
            )
            iter += 1
            if w_included:
                included += 1
                took.append(w_took)
            else:
                not_included_len.append(len(w_path))
            if w_included_pos: included_pos.append(w_included_pos)

            if iter % _WORKER_CHUNKSIZE == 0:
                percent = round((included / iter) * 100, 2)
                avg_pos = statistics.mean(included_pos)
                avg_took = statistics.mean(took)
                avg_not_included_len = statistics.mean(not_included_len or [0])
                prg.update(f'{included}/{iter} = {percent}%')

                with open(
                        _RESULT_DIR /
                        f'{now.strftime("%Y%m%d.%H%M")}-candidates-include-real.json',
                        'w', encoding='UTF-8'
                ) as f:
                    f.write(json.dumps({
                        'processed': iter,
                        'included': included,
                        'percent': percent,
                        'avg_pos': avg_pos,
                        'avg_took': avg_took,
                        'avg_not_included_len': avg_not_included_len,
                        'took': took,
                        'included_pos': included_pos,
                        'not_included_len': not_included_len
                    }, indent=2))


    prg.complete()



def _load_ris_paths() -> t.List[t.List[int]]:
    # NOTE: THESE HAVE TO BE THE SAME AS IN GET_CANDIDATES!
    # HACK: change get_candidates to be parameterizable?

    dt = datetime.fromisoformat('2023-05-01T00:00')

    def _load():
        paths = [
            p['path'] for p in itertools.chain(
                routeviews.iter_paths(dt, eliminate_path_prepending=True),
                ripe_ris.iter_paths(dt, eliminate_path_prepending=True),
            )
        ]
        random.shuffle(paths)
        return paths

    cache = PickleFileCache('research_candidates_include_real_paths', _load)

    return cache.get()



if __name__ == '__main__': _research_candidates_include_real()