from datetime import datetime
from statistics import mean
import random
import typing as t
import logging
import multiprocessing

from matplotlib import pyplot as plt
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.stage3_rank import classifier_rnn
from bgpsyche.service.ext import routeviews
from bgpsyche.stage3_rank.path_candidate_cache import PathCandidateCache
from bgpsyche.util.multiprocessing import worker_amount

_LOG = logging.getLogger(__name__)

_PREDICT_FUN = classifier_rnn.predict_probs

_WORKER_PROCESSES_AMNT = worker_amount(ram_per_worker_mb=500)
_WORKER_CHUNKSIZE = 10

_CANDIDATE_CACHE = PathCandidateCache('real')

class _PathWithProb(t.TypedDict):
    path: t.List[int]
    prob: float

class _RealLifeEvalModelWorkerResp(t.TypedDict):
    candidates_length: int
    found_position: int
    path: t.List[int]

def _real_life_eval_model_worker(path: t.List[int]) -> _RealLifeEvalModelWorkerResp:

    candidates = _CANDIDATE_CACHE.get(path[0], path[-1])
    if path not in candidates:
        _LOG.info('skipping bc not in candidates')
        return {
            'candidates_length': len(candidates),
            'found_position': -1, 'path': path,
        }

    probs: t.List[_PathWithProb] = [
        { 'path': candidates[i], 'prob': prob }
        for i, prob in enumerate(_PREDICT_FUN(candidates))
    ]
    probs.sort(key=lambda el: el['prob'], reverse=True)
    candidates_probs = [ p['path'] for p in probs ]

    return {
        'candidates_length': len(candidates_probs),
        'found_position': candidates_probs.index(path),
        'path': path,
    }


def _load_test_paths() -> t.List[t.List[int]]:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    def _load():
        paths = [
            meta['path'] for meta in
            # mrt_custom.iter_paths('mrt_dtag', eliminate_path_prepending=True,)
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-02T00:00'),
                eliminate_path_prepending=True,
                # collectors=['decix.jhb'],
            )
        ]
        random.shuffle(paths)
        return paths

    cache = PickleFileCache('research_real_life_eval_paths', _load)

    return cache.get()


def real_life_eval_model():
    real_paths = _load_test_paths()
    found_positions: t.List[float] = []
    candidate_lengths: t.List[int] = []
    iter = 0
    prg_step = 10

    _CANDIDATE_CACHE.init_caches()

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:

        iter = 0
        skipped = 0
        for w_resp in p.imap_unordered(
            _real_life_eval_model_worker, real_paths,
            chunksize=_WORKER_CHUNKSIZE
        ):
            iter += 1
            if w_resp['found_position'] == -1:
                skipped += 1
                continue

            candidate_lengths.append(w_resp['candidates_length'])
            found_positions.append(w_resp['found_position'])

            if iter % prg_step == 0:
                avg_found = round(mean(found_positions), 2)
                avg_len = round(mean(candidate_lengths))
                percent_correct = round(
                    (found_positions.count(0) / len(found_positions)) * 100
                )
                percent_correct_real = round((
                    found_positions.count(0) / (len(found_positions) + skipped)
                ) * 100)
                percent_top_3_real = round((
                    sum([found_positions.count(i) for i in [0, 1, 2]])
                    / (len(found_positions) + skipped)
                ) * 100)
                percent_skipped = round((skipped / iter) * 100)
                _LOG.info(
                    f'I: {iter} | ' +
                    f'C: {percent_correct} CR: {percent_correct_real} ' +
                    f'C3: {percent_top_3_real} | ' +
                    f'S: {percent_skipped} | ' +
                    f'Pos: {found_positions[-1]}, Avg Pos: {avg_found} | ' +
                    f'Len: {candidate_lengths[-1]}, Avg Len: {avg_len} | ' +
                    f'Top 10: {w_resp["found_position"] < 10} | ' +
                    f'{w_resp["path"]}'
                )

                plt.hist(found_positions, bins=100, range=(0, 500))
                classifier_rnn.tensorboard_writer.add_figure(
                    'eval_real/pos', plt.gcf(), iter
                )
                plt.ecdf(found_positions)
                plt.xlim([0, 50])
                classifier_rnn.tensorboard_writer.add_figure(
                    'eval_real/pos_cdf_begin', plt.gcf(), iter
                )
                plt.ecdf(found_positions)
                plt.xlim([0, 800])
                classifier_rnn.tensorboard_writer.add_figure(
                    'eval_real/pos_cdf_full', plt.gcf(), iter
                )

                for subtag, value in {
                        'percent_correct': percent_correct_real,
                        'percent_correct_top_3': percent_top_3_real,
                        'percent_correct_ignore_skipped': percent_correct,
                        'percent_skipped': percent_skipped,
                        'position': mean(found_positions[-prg_step:]),
                }.items():
                    classifier_rnn.tensorboard_writer.add_scalar(
                        f'eval_real/{subtag}', value, iter
                    )


def _main():
    _PREDICT_FUN([[3320, 3320]])

    real_life_eval_model()

if __name__ == '__main__': _main()