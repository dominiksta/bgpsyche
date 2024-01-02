from collections import defaultdict
import itertools
from math import floor
import multiprocessing
from os import cpu_count
import logging
from pprint import pformat
from statistics import mean
import typing as t
from datetime import datetime

import bgpsyche.logging_config
import numpy as np
from bgpsyche.stage2_enrich.geographic_distance import geographic_distance_diff
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.util.benchmark import Progress

_LOG = logging.getLogger(__name__)

_WORKER_SYSTEM_AVAILABLE_RAM = 16
_WORKER_RAM_ESTIMATE_GB = 4
_WORKER_PROCESSES_AMNT = floor(_WORKER_SYSTEM_AVAILABLE_RAM / _WORKER_RAM_ESTIMATE_GB)
_WORKER_CHUNKSIZE = 1_000

# config / data input definitions
# ----------------------------------------------------------------------

_ris_for_date = \
    lambda dt: ripe_ris.iter_paths(
        datetime.fromisoformat(dt), eliminate_path_prepending=True
    )
_routeviews_for_date = \
    lambda dt: routeviews.iter_paths(
        datetime.fromisoformat(dt), eliminate_path_prepending=True
    )

def _ris_routeviews_single_rib():
    return itertools.chain(
        _ris_for_date('2023-05-01T00:00'),
        _routeviews_for_date('2023-05-01T00:00'),
    )

def _ris_routeviews_week_of_ribs():
    return itertools.chain(
        _ris_for_date('2023-05-01T00:00'),
        _ris_for_date('2023-05-02T00:00'),
        _ris_for_date('2023-05-03T00:00'),
        _ris_for_date('2023-05-04T00:00'),
        _ris_for_date('2023-05-05T00:00'),
        _ris_for_date('2023-05-06T00:00'),
        _ris_for_date('2023-05-07T00:00'),
        _routeviews_for_date('2023-05-01T00:00'),
        _routeviews_for_date('2023-05-02T00:00'),
        _routeviews_for_date('2023-05-03T00:00'),
        _routeviews_for_date('2023-05-04T00:00'),
        _routeviews_for_date('2023-05-05T00:00'),
        _routeviews_for_date('2023-05-06T00:00'),
        _routeviews_for_date('2023-05-07T00:00'),
    )

_INPUT_DATA_FUN = _ris_routeviews_single_rib

# compute
# ----------------------------------------------------------------------

def _research_geographic_distance_diff():

    paths_in_mem = [ path_meta['path'] for path_meta in _INPUT_DATA_FUN() ]
    np.random.shuffle(paths_in_mem)
    prg = Progress(int(len(paths_in_mem) / _WORKER_CHUNKSIZE), 'total')

    iter = 0
    distances: t.List[float] = []
    meta = defaultdict(int)

    # HACK: populate cache before invididual processes try to
    geographic_distance_diff([3320, 3320])

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        for res in p.imap_unordered(
                geographic_distance_diff, paths_in_mem,
                chunksize=_WORKER_CHUNKSIZE,
        ):
            w_diff, w_meta_currs = res
            iter += 1

            for meta_curr in w_meta_currs: meta[meta_curr] += 1
            distances.append(w_diff)

            if iter % _WORKER_CHUNKSIZE == 0:
                prg.update()
                _LOG.info(
                    f'Current result: Avg distance: {round(mean(distances))}km ' +
                    f'({iter}, {pformat(dict(meta))})'
                )


    prg.complete()
    _LOG.info(
        f'Result: Avg distance: {round(mean(distances))}km ' +
        f'({iter}, {pformat(dict(meta))})'
    )


if __name__ == '__main__': _research_geographic_distance_diff()