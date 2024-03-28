from itertools import chain
import logging
from random import shuffle
import signal
from types import FrameType
import typing as t
from datetime import datetime
import statistics

import editdistance
import networkx as nx
from matplotlib import pyplot as plt
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.logging_config import logging_setup
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates

logging_setup()
_LOG = logging.getLogger(__name__)

_MAX_CANDIDATES = 100

def _load_paths() -> t.List[t.List[int]]:
    paths = [
        path_meta['path'] for path_meta in
        chain(
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00'),
                eliminate_path_prepending=True,
            )
        )
    ]
    shuffle(paths)
    return paths


def _research_compute_candidates_len_diff():
    edit_distances: t.List[float] = []
    candidates_amount: t.List[int] = []
    len_diffs: t.List[int] = []
    found = 0
    missing_node = 0
    iter = 0

    cancel = False
    sigint_orig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    def sigint_handler(sig: int, frame: t.Optional[FrameType]):
        nonlocal cancel, sigint_orig_handler
        _LOG.warning('Training cancelled because SIGINT (Ctrl+C)')
        signal.signal(signal.SIGINT, sigint_orig_handler)
        cancel = True
    signal.signal(signal.SIGINT, sigint_handler)

    def _show_progress():
        _LOG.info(f'iter      : {iter}')
        _LOG.info(f'dist_mean : {statistics.mean(edit_distances)}')
        _LOG.info(f'amount    : {statistics.mean(candidates_amount)}')
        _LOG.info(f'found     : {found / iter}')
        _LOG.info(f'missing_n : {missing_node / iter}')

    for p in _load_paths():
        if cancel: break
        iter += 1

        try:
            candidates = get_path_candidates(p[0], p[-1])[:_MAX_CANDIDATES]
            candidates_amount.append(len(candidates))
            if p in candidates: found += 1
            edit_distances.extend([ editdistance.eval(p, can) for can in candidates ])
            len_diffs.extend([ len(can) - len(p) for can in candidates ])
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            missing_node += 1

        if iter % 100 == 0: _show_progress()

    _show_progress()

    _ = plt.figure(1)
    plt.ecdf(edit_distances)

    _ = plt.figure(2)
    diffs = [-3, -2, -1, 0, 1, 2, 3]
    plt.bar(diffs, [ len_diffs.count(d) / iter for d in diffs ])

    plt.show()


if __name__ == '__main__': _research_compute_candidates_len_diff()
