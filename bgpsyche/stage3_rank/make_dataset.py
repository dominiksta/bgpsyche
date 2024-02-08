from datetime import datetime
import logging
import multiprocessing
from os import cpu_count
from types import FrameType
import typing as t
import signal

import numpy as np
from bgpsyche.caching.json import JSONFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.stage1_candidates.get_candidates import (
    abort_on_amount, abort_on_timeout, get_path_candidates
)
from bgpsyche.stage2_enrich.enrich import enrich_asn, enrich_path
from bgpsyche.stage3_rank.vectorize_features import (
    vectorize_as_features, vectorize_path_features
)
from bgpsyche.util.benchmark import Progress
from bgpsyche.service.ext import routeviews, ripe_ris
from bgpsyche.util.run_in_pypy import run_in_pypy

logging_setup()
_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (cpu_count() or 3) - 3
_WORKER_CHUNKSIZE = 10

def _get_path_candidates_worker(path: t.List[int]):
    return get_path_candidates(
        source=path[0], sink=path[-1],
        abort_on=lambda: [
            # since we really only want to find a handful of wrong paths, it
            # does not matter if we find the correct path before the timeout.
            { 'func': abort_on_timeout(0.7), 'desc': 'timeout .7s' },
            # TODO: set to 200 ^^20231221-133927 Research Log_ BGPsyche Candidate Search^^
            { 'func': abort_on_amount(200), 'desc': 'found 50' },
        ],
        # quiet=True,
    ), path


def _iter_path_with_alternatives(
        candidates_per_real_path: int,
        real_paths_n: int,
        routeviews_dts: t.List[str],
        ripe_ris_dts: t.List[str],
        progress_msg = 'Preparing Dataset',
) -> t.Iterator[t.Tuple[t.List[int], t.List[t.List[int]]]]:
    real_paths: t.List[t.List[int]] = []

    _LOG.info('Loading paths into memory...')

    for dt in routeviews_dts:
        for path_meta in routeviews.iter_paths(
                datetime.fromisoformat(dt), eliminate_path_prepending=True
        ):
            real_paths.append(path_meta['path'])

    for dt in ripe_ris_dts:
        for path_meta in ripe_ris.iter_paths(
                datetime.fromisoformat(dt), eliminate_path_prepending=True
        ):
            real_paths.append(path_meta['path'])
            

    _LOG.info('Done loading paths into memory')
    _LOG.info(f'Shuffling paths and taking {real_paths_n}')

    np.random.shuffle(real_paths)
    real_paths = real_paths[:real_paths_n]

    prg_len = 5
    prg = Progress(int(len(real_paths) / prg_len), progress_msg)

    # HACK: initialize cache in main process
    get_path_candidates(3320, 3320)


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
                np.random.shuffle(resp['candidates'])
                out_candidates = []
                for candidate in resp['candidates']:
                    if candidate == path: continue
                    if len(out_candidates) >= candidates_per_real_path: break
                    out_candidates.append(candidate)

                yield path, out_candidates

                if iter % prg_len == 0: prg.update()
        except KeyboardInterrupt:
            _LOG.warning('Make dataset cancelled because SIGINT (Ctrl+C)')
            p.terminate()
            p.join()

        prg.complete()


# path level
# ----------------------------------------------------------------------

class DatasetPathLevelEl(t.TypedDict):
    real: bool
    path: t.List[int]
    path_features: t.List[t.Union[int, float]]
    as_features: t.List[t.List[t.Union[int, float]]]

@run_in_pypy(cache=JSONFileCache)
def make_dataset(
        candidates_per_real_path = 1,
        real_paths_n = 1_000,
        routeviews_dts: t.List[str] = [
            '2023-05-01T00:00',
        ],
        ripe_ris_dts: t.List[str] = [
            '2023-05-01T00:00',
        ],
) -> t.List[DatasetPathLevelEl]:
    out: t.List[DatasetPathLevelEl] = []
    for real, alternatives in _iter_path_with_alternatives(
            candidates_per_real_path=candidates_per_real_path,
            real_paths_n=real_paths_n,
            routeviews_dts=routeviews_dts,
            ripe_ris_dts=ripe_ris_dts,
    ):
        for alternative in alternatives:
            out.append({
                'path_features': vectorize_path_features(enrich_path(alternative)),
                'as_features': [
                    vectorize_as_features(enrich_asn(asn)) for asn in alternative
                ],
                'path': alternative, 'real': False,
            })

        out.append({
            'path_features': vectorize_path_features(enrich_path(real)),
            'as_features': [
                vectorize_as_features(enrich_asn(asn)) for asn in real
            ],
            'path': real, 'real': True,
        })

    return out