import functools
import itertools
import logging
from time import time
import typing as t
from datetime import datetime

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext import ripe_ris, routeviews

_LOG = logging.getLogger(__name__)

class PathCandidatesRes(t.TypedDict):
    candidates: t.List[t.List[int]]
    by_length: t.Dict[int, t.List[t.List[int]]]


PathCandidatesByLen = t.Dict[int, t.List[t.List[int]]]


def abort_on_timeout(timeout_s: float) -> t.Callable[[t.List[int]], bool]:
    before = time()
    return lambda _path: (time() - before) > timeout_s


def get_path_candidates(
        source: int, sink: int,
        abort_on: t.List[t.Callable[[t.List[int]], bool]] = [],
        quiet: bool = False,
) -> PathCandidatesRes:
    if not quiet: _LOG.info(f'Getting Path Candidates {source}->{sink}')
    as_graph = _get_as_graph()

    candidates = _candidates_all_paths_starting_shortest(
        as_graph, source, sink,
        timeout_s=5, max_paths=4000,
        abort_on=abort_on, quiet=quiet
    )

    return {
        'candidates': candidates,
        'by_length': _mk_paths_dict_by_length(candidates),
    }


def flatten_candidates(
        candidates: PathCandidatesByLen
) -> t.List[t.List[int]]:
    return list(itertools.chain.from_iterable(candidates.values()))


def _candidates_all_paths_starting_shortest(
        as_graph: nx.Graph, source: int, sink: int,
        timeout_s: float, max_paths: int,
        abort_on: t.List[t.Callable[[t.List[int]], bool]] = [],
        quiet: bool = False,
) -> t.List[t.List[int]]:
    time_start = time()
    ret: t.List[t.List[int]] = []
    iter = 0
    for path in nx.shortest_simple_paths(as_graph, source, sink):
        ret.append(path)

        iter += 1
        if iter % 500 == 0:
            # _LOG.info(f'Found {iter} paths')
            if time() - time_start > timeout_s:
                if not quiet: _LOG.info(
                    f'Path search timed out after {timeout_s}s ' +
                    f' with {iter} paths found'
                )
                break

        if True in ( func(path) for func in abort_on ):
            _LOG.debug(
                f'Path search finished bc abort callback returned True'
            )
            break

        if iter >= max_paths:
            _LOG.debug(
                f'Path search finished with {iter} paths found'
            )
            break

    return ret


def _mk_paths_dict_by_length(
        paths: t.List[t.List[int]]
) -> t.Dict[int, t.List[t.List[int]]]:
    ret: t.Dict[int, t.List[t.List[int]]] = {}
    for path in paths:
        l = len(path)
        if l not in ret: ret[l] = []
        ret[l].append(path)
    return ret


@functools.lru_cache()
def _get_as_graph() -> nx.Graph:
    as_graph_cache = PickleFileCache(
        'as_graphs',
        lambda: as_graphs_from_paths(itertools.chain(
            routeviews.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00')
            ),
            ripe_ris.iter_paths(
                datetime.fromisoformat('2023-05-01T00:00')
            ),
        ))[0]
    )

    # as_graph_cache.invalidate()
    return as_graph_cache.get()

