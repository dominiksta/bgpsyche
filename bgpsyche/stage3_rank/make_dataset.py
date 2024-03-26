from datetime import datetime
from itertools import pairwise
import logging
import multiprocessing
from os import cpu_count
import typing as t
import signal

import numpy as np
from bgpsyche.caching.json import JSONFileCache
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.stage1_candidates.get_candidates import (
    abort_on_amount, abort_on_timeout
)
from bgpsyche.stage2_enrich.enrich import enrich_asn, enrich_link, enrich_path
from bgpsyche.stage3_rank.path_candidate_cache import PathCandidateCache
from bgpsyche.stage3_rank.vectorize_features import (
    vectorize_as_features, vectorize_link_features, vectorize_path_features
)
from bgpsyche.service.ext import routeviews, ripe_ris
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (cpu_count() or 3) - 3
_WORKER_CHUNKSIZE = 10

_CANDIDATE_CACHE = PathCandidateCache(
    'make_dataset_less_candidates',
    abort_customer_cone_search=lambda: [
        { 'func': abort_on_timeout(0.7), 'desc': 'timeout .7s' },
        { 'func': abort_on_amount(10), 'desc': 'found 200' },
    ],
    abort_full_search=lambda: [
        { 'func': abort_on_timeout(0.7), 'desc': 'timeout .7s' },
        { 'func': abort_on_amount(10), 'desc': 'found 200' },
    ],
    quiet=True,
)


@PickleFileCache.decorate
def _load_paths(
        routeviews_dts: t.List[str],
        ripe_ris_dts: t.List[str],
        n: int,
) -> t.List[t.List[int]]:
    ret: t.List[t.List[int]] = []
    _LOG.info('Loading paths into memory...')

    for dt in routeviews_dts:
        for path_meta in routeviews.iter_paths(
                datetime.fromisoformat(dt), eliminate_path_prepending=True
        ):
            ret.append(path_meta['path'])

    for dt in ripe_ris_dts:
        for path_meta in ripe_ris.iter_paths(
                datetime.fromisoformat(dt), eliminate_path_prepending=True
        ):
            ret.append(path_meta['path'])


    _LOG.info('Done loading paths into memory')
    _LOG.info(f'Shuffling paths and taking {n}')

    np.random.shuffle(ret)
    ret = ret[:n]

    return ret


def _get_path_candidates_worker(path: t.List[int]):
    return _CANDIDATE_CACHE.get(path[0], path[-1]), path


def _iter_path_with_alternatives(
        candidates_per_real_path: int,
        real_paths_n: int,
        routeviews_dts: t.List[str],
        ripe_ris_dts: t.List[str],
        progress_msg = 'Preparing Dataset',
) -> t.Iterator[t.Tuple[t.List[int], t.List[t.List[int]]]]:
    real_paths= _load_paths(routeviews_dts, ripe_ris_dts, real_paths_n)

    prg_len = 20
    prg = Progress(int(len(real_paths) / prg_len), progress_msg)

    _CANDIDATE_CACHE.init_caches()

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        signal.signal(signal.SIGINT, original_sigint_handler)
        iter = 0
        try:
            for w_resp in p.imap_unordered(
                    _get_path_candidates_worker, real_paths,
                    chunksize=_WORKER_CHUNKSIZE
            ):
                iter += 1
                resp, path = w_resp
                resp.sort(key=len)
                # np.random.shuffle(resp)
                out_candidates = []
                for candidate in resp:
                    if candidate == path: continue
                    if len(out_candidates) >= candidates_per_real_path: break
                    out_candidates.append(candidate)

                assert len(out_candidates) == 0 or \
                    len(out_candidates[0]) <= len(out_candidates[-1])
                yield path, out_candidates

                if iter % prg_len == 0: prg.update()
        except KeyboardInterrupt:
            _LOG.warning('Make dataset cancelled because SIGINT (Ctrl+C)')
            p.terminate()
            p.join()

        prg.complete()


class DatasetEl(t.TypedDict):
    real: bool
    path: t.List[int]
    path_features : t.List[t.Union[int, float]]
    link_features : t.List[t.List[t.Union[int, float]]]
    as_features   : t.List[t.List[t.Union[int, float]]]

@run_in_pypy(cache=JSONFileCache)
def make_dataset(
        candidates_per_real_path = 5,
        real_paths_n = 100_000,
        routeviews_dts: t.List[str] = [
            '2023-05-01T00:00',
        ],
        ripe_ris_dts: t.List[str] = [
            '2023-05-01T00:00',
        ],
) -> t.List[DatasetEl]:
    out: t.List[DatasetEl] = []

    def make_single_element(real: bool, path: t.List[int]) -> DatasetEl:
        return {
            'path_features': vectorize_path_features(enrich_path(path)),
            'as_features': [
                vectorize_as_features(enrich_asn(asn)) for asn in path
            ],
            'link_features': [
                vectorize_link_features(enrich_link(source, sink))
                    for source, sink in pairwise(path)
            ],
            'path': path,
            'real': real,
        }

    for real, alternatives in _iter_path_with_alternatives(
            candidates_per_real_path=candidates_per_real_path,
            real_paths_n=real_paths_n,
            routeviews_dts=routeviews_dts,
            ripe_ris_dts=ripe_ris_dts,
    ):
        if len(alternatives) == 0:
            _LOG.warning(f'No alternatives found for path {real}, skipping')
            continue

        for alternative in alternatives:
            out.append(make_single_element(real=False, path=alternative))

        out.append(make_single_element(real=True, path=real))

    return out