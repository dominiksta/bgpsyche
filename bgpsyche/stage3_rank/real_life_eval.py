from statistics import mean
import typing as t
import logging

import numpy as np

from bgpsyche.stage1_candidates.get_candidates import get_path_candidates

_LOG = logging.getLogger(__name__)

class _PathWithProb(t.TypedDict):
    path: t.List[int]
    prob: float


# def _real_life_eval_model_worker(args):
#     path: t.List[int] = args[0]


def real_life_eval_model(
        real_paths: t.List[t.List[int]],
        predict_probs: t.Callable[
            [t.List[t.List[int]]],
            t.List[float]
        ],
):
    np.random.shuffle(real_paths)
    found_positions: t.List[float] = []
    candidate_lengths: t.List[int] = []
    iter = 0
    for path in real_paths:
        iter += 1

        candidates = list(get_path_candidates(path[0], path[-1])['candidates'])
        if path not in candidates:
            _LOG.info('skipping bc not in candidates')
            continue

        probs: t.List[_PathWithProb] = [
            { 'path': candidates[i], 'prob': prob }
            for i, prob in enumerate(predict_probs(candidates))
        ]
        probs.sort(key=lambda el: el['prob'], reverse=True)
        candidates_probs = [ p['path'] for p in probs ]

        candidate_lengths.append(len(candidates_probs))
        found_positions.append(candidates_probs.index(path))

        avg_found = round(mean(found_positions), 2)
        avg_len = round(mean(candidate_lengths))
        _LOG.info(
            f'Pos: {found_positions[-1]}, Avg Pos: {avg_found} | ' +
            f'Len: {candidate_lengths[-1]}, Avg Len: {avg_len} | ' +
            f'Top 10: {path in candidates_probs[:10]} | ' +
            f'{path}'
        )