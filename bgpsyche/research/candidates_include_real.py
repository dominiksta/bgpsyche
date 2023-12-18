import itertools
import multiprocessing
import json
import typing as t
from datetime import datetime
import logging
import statistics
import os

import bgpsyche.logging_config
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import HERE

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (os.cpu_count() or 3) - 2
_WORKER_CHUNKSIZE = 100
_RESULT_DIR = HERE / 'research' / 'results'

def _research_candidates_include_real_worker(args) -> t.Tuple[bool, t.Optional[int]]:
    path: t.List[int] = args[0]
    candidates = get_path_candidates(
        path[0], path[-1],
        abort_on=[ lambda p: p == path ],
        quiet=True,
    )['candidates']
    if path in candidates:
        return True, candidates.index(path)
    else:
        return False, None


def _research_candidates_include_real():
    ris_paths = _load_ris_paths()

    # HACK: initialize cache before workers all start populating cache
    get_path_candidates(3320, 3320)

    worker_params = ( (path,) for path in ris_paths )

    iter, included, included_pos = 0, 0, []

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
            w_included, w_included_pos = t.cast(
                t.Tuple[bool, t.Optional[int]], res
            )
            iter += 1
            if w_included: included += 1
            if w_included_pos: included_pos.append(w_included_pos)

            if iter % _WORKER_CHUNKSIZE == 0:
                prg.update()
                percent = round((included / iter) * 100, 2)
                avg_pos = statistics.mean(included_pos)
                _LOG.info(
                    f'Current result: {percent} ({included}/{iter}), ' +
                    f'avg pos: {avg_pos}'
                )

                with open(
                        _RESULT_DIR /
                        f'{now.strftime("%Y%m%d.%H%M")}-candidates-include-real.json',
                        'w', encoding='UTF-8'
                ) as f:
                    f.write(json.dumps({
                        'percent': percent,
                        'avg_pos': avg_pos,
                        'included_pos': included_pos,
                    }))


    prg.complete()



def _load_ris_paths() -> t.List[t.List[int]]:
    # NOTE: THESE HAVE TO BE THE SAME AS IN GET_CANDIDATES!
    # HACK: change get_candidates to be parameterizable?

    return [
        p['path'] for p in itertools.chain(
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00')
            ),
            ripe_ris.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00')
            ),
        )
    ]


if __name__ == '__main__': _research_candidates_include_real()