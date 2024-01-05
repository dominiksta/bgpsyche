import itertools
import multiprocessing
import json
from time import time
import typing as t
from datetime import datetime
import logging
import statistics
from os import cpu_count

import bgpsyche.logging_config
import numpy as np
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import HERE

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (cpu_count() or 3) - 2
_WORKER_CHUNKSIZE = 100
_RESULT_DIR = HERE / 'research' / 'results'

def _research_candidates_include_real_worker(args) -> t.Tuple[
        bool, t.Optional[int], t.List[int], float
]:
    path: t.List[int] = args[0]
    before = time()
    candidates = get_path_candidates(
        path[0], path[-1],
        abort_on=[{ 'func': lambda p: p == path, 'desc': 'path eq' }],
        quiet=True,
    )['candidates']
    if path in candidates:
        return True, candidates.index(path), path, round(time() - before, 2)
    else:
        # _LOG.info(f'Path not found: {path}')
        return False, None, path, 0.0


def _research_candidates_include_real():
    ris_paths = _load_ris_paths()
    np.random.shuffle(ris_paths)

    # HACK: initialize cache before workers all start populating cache
    get_path_candidates(3320, 3320)

    worker_params = ( (path,) for path in ris_paths )

    iter, included, included_pos, not_included_len, took = 0, 0, [], [], []

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
                prg.update()
                percent = round((included / iter) * 100, 2)
                avg_pos = statistics.mean(included_pos)
                avg_took = statistics.mean(took)
                avg_not_included_len = statistics.mean(not_included_len or [0])
                _LOG.warning(
                    f'Current result: {percent} ({included}/{iter}), ' +
                    f'avg pos: {avg_pos}, avg took: {avg_took}, ' +
                    f'avg not_included_len: {avg_not_included_len}'
                )

                with open(
                        _RESULT_DIR /
                        f'{now.strftime("%Y%m%d.%H%M")}-candidates-include-real.json',
                        'w', encoding='UTF-8'
                ) as f:
                    f.write(json.dumps({
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

    return [
        p['path'] for p in itertools.chain(
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00'),
                eliminate_path_prepending=True,
            ),
            ripe_ris.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00'),
                eliminate_path_prepending=True,
            ),
        )
    ]


if __name__ == '__main__': _research_candidates_include_real()