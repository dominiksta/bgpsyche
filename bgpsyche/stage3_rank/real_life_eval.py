from datetime import date, datetime
from copy import deepcopy
from pprint import pprint
from statistics import mean
import random
import typing as t
import logging
import multiprocessing

import matplotlib
matplotlib.use('Agg') # dont render an invisible tk window (which does not play
                      # well with multiprocessing)
from matplotlib import pyplot as plt
import editdistance
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates
from bgpsyche.util.bgp.valley_free import path_is_valley_free
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.stage3_rank import classifier_rnn, classifier, classifier_nn
from bgpsyche.service.ext import routeviews
from bgpsyche.stage3_rank.path_candidate_cache import PathCandidateCache
from bgpsyche.util.multiprocessing import worker_amount
from bgpsyche.stage3_rank.tensorboard import tensorboard_writer as tsw

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = worker_amount(ram_per_worker_mb=500)
_WORKER_CHUNKSIZE = 10

_CANDIDATE_CACHE = PathCandidateCache('real')

_CANDIDATES_USE_FIRST_N = 1000

_STOP_AFTER = 1_000

# choose prediction function
# ----------------------------------------------------------------------

_PredictFun = t.Callable[[t.List[t.List[int]]], t.Dict[str, float]]

def _dist_from_ordered_list(
        paths: t.List[t.List[int]],
        sorted: t.List[t.List[int]],
) -> t.Dict[str, float]:
    l = len(paths)
    return { str(path): 1 - sorted.index(path) / l for path in paths }


_is_vf = lambda p: int(not path_is_valley_free(
    get_caida_asrel(date.fromisoformat('2023-05-01')), p
))

# def _predict_fun_shortest(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
#     _paths = paths.copy()
#     random.shuffle(_paths)
#     s = sorted(_paths, key=len)
#     return _dist_from_ordered_list(_paths, s)
# _PREDICT_FUN: _PredictFun = _predict_fun_shortest

# def _predict_fun_shuffle(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
#     s = deepcopy(paths)
#     random.shuffle(s)
#     return _dist_from_ordered_list(paths, s)
# _PREDICT_FUN: _PredictFun = _predict_fun_shuffle

# def _predict_fun_shortest_vf(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
#     s = sorted(paths, key=lambda p: (len(p), _is_vf(p)))
#     return _dist_from_ordered_list(paths, s)
# _PREDICT_FUN: _PredictFun = _predict_fun_shortest_vf

# def _predict_fun_vf_shortest(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
#     is_vf = lambda p: int(not path_is_valley_free(
#         get_caida_asrel(date.fromisoformat('2023-05-01')), p
#     ))
#     s = sorted(paths, key=lambda p: (is_vf(p), len(p)))
#     return _dist_from_ordered_list(paths, s)

# _PREDICT_FUN: _PredictFun = _predict_fun_vf_shortest


_nn_predict = classifier_nn.make_prediction_function(retrain=True)


# def _predict_fun_shortest_sort_by_nn(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
#     probs = _nn_predict(paths)
#     path_probs = { str(path): prob for prob, path in zip(probs, paths) }

#     s = sorted(paths, key=lambda p: (len(p), 1 - path_probs[str(p)]))
#     return _dist_from_ordered_list(paths, s)

# _PREDICT_FUN: _PredictFun = _predict_fun_shortest_sort_by_nn

def _predict_fun_nn(paths: t.List[t.List[int]]) -> t.Dict[str, float]:
    probs = _nn_predict(paths)
    path_probs = { str(path): prob for prob, path in zip(probs, paths) }

    s = sorted(paths, key=lambda p: (1 - path_probs[str(p)], len(p)))
    return _dist_from_ordered_list(paths, s)

_PREDICT_FUN: _PredictFun = _predict_fun_nn


# _PREDICT_FUN: _PredictFun = classifier_rnn.predict_probs
# _PREDICT_FUN: _PredictFun = classifier_nn.make_prediction_function()
# _PREDICT_FUN: _PredictFun = classifier.predict_probs

# ----------------------------------------------------------------------

class _PathWithProb(t.TypedDict):
    path: t.List[int]
    prob: float

class _RealLifeEvalModelWorkerResp(t.TypedDict):
    candidates: t.List[t.List[int]]
    path: t.List[int]
    probs: t.Dict[str, float]

def _real_life_eval_model_worker(path: t.List[int]) -> _RealLifeEvalModelWorkerResp:
    candidates = _CANDIDATE_CACHE.get(path[0], path[-1])
    random.shuffle(candidates)
    # candidates = get_path_candidates(path[0], path[-1])
    candidates = sorted(candidates, key=len)[:_CANDIDATES_USE_FIRST_N]
    # assert len(candidates[0]) <= len(candidates[-1])
    random.shuffle(candidates)
    # candidates.sort(key=len)
    probs = _PREDICT_FUN(candidates)
    ret: _RealLifeEvalModelWorkerResp = \
        { 'candidates': candidates, 'path': path, 'probs': probs }
    # pprint(ret)
    return ret


def _load_test_paths() -> t.List[t.List[int]]:
    def _load():
        paths = [
            meta['path'] for meta in
            # mrt_custom.iter_paths('mrt_dtag', eliminate_path_prepending=True,)
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00'),
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
    edit_distances: t.List[int] = []
    len_diff: t.List[int] = []
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
            if iter >= _STOP_AFTER:
                _LOG.warn(f'Stopping after {_STOP_AFTER} iterations')
                break

            path = w_resp['path']

            if path not in w_resp['candidates']:
                skipped += 1
                continue

            candidates_probs = sorted(
                w_resp['candidates'], key=lambda p: 1 - w_resp['probs'][str(p)]
            )
            # pprint(candidates_probs)

            path = w_resp['path']
            candidates_length = len(candidates_probs)

            found_position = candidates_probs.index(path)

            edit_distances.append(editdistance.eval(path, candidates_probs[0]))
            len_diff.append(len(path) - len(candidates_probs[0]))
            candidate_lengths.append(candidates_length)
            found_positions.append(found_position)

            if iter % prg_step == 0:
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
                    f'Avg Pos: {mean(found_positions):.1f} | ' +
                    f'Avg Len: {mean(candidate_lengths):.1f} | ' +
                    f'Top 10: {found_position < 10} | ' +
                    f'{w_resp["path"]}'
                )

                plt.hist(found_positions, bins=100, range=(0, 500))
                tsw.add_figure('eval_real/pos', plt.gcf(), iter)

                plt.ecdf(found_positions)
                plt.xlim([0, 50])
                tsw.add_figure('eval_real/pos_cdf_begin', plt.gcf(), iter)
                plt.ecdf(found_positions)
                plt.xlim([0, 100])
                tsw.add_figure('eval_real/pos_cdf_begin_100', plt.gcf(), iter)
                plt.ecdf(found_positions)
                plt.xlim([0, 800])
                tsw.add_figure('eval_real/pos_cdf_full', plt.gcf(), iter)

                found_in_first_n_percent = [
                    round((found_positions[i] / candidate_lengths[i]) * 100)
                    for i in range(len(found_positions))
                ]

                plt.ecdf(found_in_first_n_percent)
                plt.xlim([0, 100])
                tsw.add_figure('eval_real/pos_percent_cdf_100', plt.gcf(), iter)

                plt.ecdf(edit_distances)
                plt.xlim([0, 10])
                tsw.add_figure('eval_real/edit_distance_cdf', plt.gcf(), iter)
                plt.ecdf(len_diff)
                plt.xlim([0, 10])
                tsw.add_figure('eval_real/len_diff', plt.gcf(), iter)

                # TODO: when an error happens, does it happen in the front,
                # middle or end of the path?

                for subtag, value in {
                        'percent_correct': percent_correct_real,
                        'percent_correct_top_3': percent_top_3_real,
                        'percent_correct_ignore_skipped': percent_correct,
                        'percent_skipped': percent_skipped,
                        'position': mean(found_positions),
                        'in_first_percent': mean(found_in_first_n_percent),
                        'edit_distance': mean(edit_distances),
                }.items():
                    tsw.add_scalar(f'eval_real/{subtag}', value, iter)


def _main():
    _PREDICT_FUN([[3320, 3320]])

    real_life_eval_model()

if __name__ == '__main__': _main()