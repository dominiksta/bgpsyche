from datetime import datetime
import logging
import multiprocessing
from os import cpu_count
import typing as t

import numpy as np
import bgpsyche.logging_config
from bgpsyche.stage1_candidates.get_candidates import (
    abort_on_amount, abort_on_timeout, get_path_candidates
)
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.vectorize_features import vectorize_features
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import JOBLIB_MEMORY
from bgpsyche.service.ext import routeviews, ripe_ris

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (cpu_count() or 3) - 2
_WORKER_CHUNKSIZE = 10


def _get_path_candidates_worker(path: t.List[int]):
    return get_path_candidates(
        source=path[0], sink=path[-1],
        abort_on=[
            # since we really only want to find a handful of wrong paths, it
            # does not matter if we find the correct path before the timeout.
            { 'func': abort_on_timeout(0.7), 'desc': 'timeout .7s' },
            { 'func': abort_on_amount(50), 'desc': 'found 50' },
        ],
        unordered=True
    ), path


@JOBLIB_MEMORY.cache
def make_path_dataset(
        candidates_per_real_path = 10,
        real_paths_n = 10_000,
        routeviews_dts: t.List[datetime] = [
            datetime.fromisoformat('2023-05-01T00:00'),
        ],
        ripe_ris_dts: t.List[datetime] = [
            datetime.fromisoformat('2023-05-01T00:00'),
        ],
) -> t.Tuple[np.ndarray, np.ndarray]:
    real_paths: t.List[t.List[int]] = []
    X, y = [], []

    _LOG.info('Loading paths into memory...')

    for dt in routeviews_dts:
        for path_meta in routeviews.iter_paths(dt, eliminate_path_prepending=True):
            real_paths.append(path_meta['path'])

    for dt in ripe_ris_dts:
        for path_meta in ripe_ris.iter_paths(dt, eliminate_path_prepending=True):
            real_paths.append(path_meta['path'])
            

    _LOG.info('Done loading paths into memory')
    _LOG.info(f'Shuffling paths and taking {real_paths_n}')

    np.random.shuffle(real_paths)
    real_paths = real_paths[:real_paths_n]

    prg = Progress(
        int(len(real_paths) / _WORKER_CHUNKSIZE),
        'Mixing real paths with false candidates'
    )

    # HACK: initialize cache in main process
    get_path_candidates(3320, 3320)

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        iter = 0
        for w_resp in p.imap_unordered(
                _get_path_candidates_worker, real_paths,
                chunksize=_WORKER_CHUNKSIZE
        ):
            iter += 1
            resp, path = w_resp
            np.random.shuffle(resp['candidates'])
            iter_candidates = 0
            for candidate in resp['candidates']:
                if candidate == path: continue
                iter_candidates += 1
                if iter_candidates >= candidates_per_real_path: break
                X.append(vectorize_features(enrich_path(candidate)))
                y.append(0)

            X.append(vectorize_features(enrich_path(path)))
            y.append(1)

            if iter % _WORKER_CHUNKSIZE == 0:
                prg.update()

        prg.complete()


    return np.array(X), np.array(y)
        

if __name__ == '__main__': make_path_dataset()