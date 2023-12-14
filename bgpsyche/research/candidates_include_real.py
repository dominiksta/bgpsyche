import multiprocessing
import json
import typing as t
from datetime import datetime
import logging
import statistics

import numpy as np
import bgpsyche.logging_config
from bgpsyche.stage1_candidates import get_path_candidates, flatten_candidates
from bgpsyche.service.ext import ripe_ris
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import HERE

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = 6
_RESULT_DIR = HERE / 'research' / 'results'

def _research_candidates_include_real_worker(args) -> t.Tuple[int, int, t.List[int]]:
    paths: t.List[t.List[int]] = args[0]
    worker_num: int = args[1]
    # queue: multiprocessing.Queue = args[2]

    prg = Progress(len(paths), f'worker_{worker_num}')

    iter = 0
    included = 0
    included_pos: t.List[int] = []
    for path in paths:
        iter += 1; prg.update()
        candidates = flatten_candidates(get_path_candidates(path[0], path[-1]))
        if path in candidates:
            included += 1
            included_pos.append(candidates.index(path))

        if iter % 1 == 0:
            percent = round((included / iter) * 100, 2)
            _LOG.warning(
                f'Worker {worker_num} result: {percent}% - {included}/{iter}'
            )
            # queue.put((iter, included))

    return iter, included, included_pos


# def _worker_watcher(queue: multiprocessing.Queue) -> None:
#     _LOG.info('Starting Watcher Process')
#     while True:
#         msg = queue.get()
#         if msg == 'stop':
#             _LOG.info('Stopping Watcher Process')
#             break
#         iter, included = t.cast(t.Tuple[int, int], msg)
#         percent = round((included / iter) * 100, 2)
#         _LOG.warning(f'Current result: {percent}% of {iter}')


def _research_candidates_include_real():
    ris_paths = _load_ris_paths()

    paths_split_by_worker: t.List[t.List[int]] = [
        list(a) for a in np.array_split(ris_paths, _WORKER_PROCESSES_AMNT)
    ]

    worker_params = []
    # queue = multiprocessing.Queue()

    for i in range(0, _WORKER_PROCESSES_AMNT):
        worker_params.append((paths_split_by_worker[i], i))

    iter, included, included_pos = 0, 0, []

    # watcher = multiprocessing.Process(
    #     target=_worker_watcher, args=(queue,)
    # )
    # watcher.start()

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        for res in p.imap_unordered(
                _research_candidates_include_real_worker,
                worker_params
        ):
            w_iter, w_included, w_included_pos = t.cast(
                t.Tuple[int, int, t.List[int]], res
            )
            iter += w_iter
            included += w_included
            included_pos += w_included_pos
            percent = round((included / iter) * 100, 2)
            avg_pos = statistics.mean(w_included_pos)
            _LOG.warning(f'Worker Done: {percent}, avg pos: {avg_pos}')

    percent = round((included / iter) * 100, 2)
    avg_pos = statistics.mean(included_pos)

    with open(
            _RESULT_DIR /
            f'{datetime.now().strftime("%Y%m%d.%H%M")}-candidates-include-real.json',
            'w', encoding='UTF-8'
    ) as f:
        f.write(json.dumps({
            'percent': percent,
            'avg_pos': avg_pos,
            'included_pos': included_pos,
        }))

    _LOG.warning(f'All Done: {percent}, avg pos: {avg_pos}')
    # queue.put('stop')



def _load_ris_paths() -> t.List[t.List[int]]:
    return [
        p['path'] for p in ripe_ris.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00')
        )
    ]


if __name__ == '__main__': _research_candidates_include_real()