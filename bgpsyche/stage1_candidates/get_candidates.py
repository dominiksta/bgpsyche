import functools
import itertools
import logging
from time import time
import typing as t
from datetime import datetime

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths, bgp_graph_to_networkx
from bgpsyche.service.ext import ripe_ris, routeviews

_LOG = logging.getLogger(__name__)

PathCandidatesByLen = t.Dict[int, t.List[t.List[int]]]

class AllPathCandidates(t.TypedDict):
    shortest: PathCandidatesByLen
    all: PathCandidatesByLen

def get_path_candidates(
        source: int, sink: int
) -> AllPathCandidates:
    _LOG.info(f'Getting Path Candidates {source}->{sink}')
    as_graph = _get_as_graph()

    candidates_shortest = _mk_paths_dict_by_length(
        _candidates_shortest_paths(as_graph, source, sink)
    )

    candidates_all = _mk_paths_dict_by_length(
        _candidates_all_paths_starting_shortest(
            as_graph, source, sink,
            timeout_s=20, max_paths=4000
        )
    )

    return {
        'shortest': candidates_shortest,
        'all': candidates_all,
    }


def flatten_candidates(
        candidates: AllPathCandidates
) -> t.List[t.List[int]]:
    ret: t.List[t.List[int]] = []
    for kind in candidates.keys():
        for length in candidates[kind].keys():
            ret.extend(candidates[kind][length])
    return ret


def _candidates_shortest_paths(
        as_graph: nx.DiGraph, source: int, sink: int
) -> t.List[t.List[int]]:
    _LOG.info(f'Getting Shortest Paths {source}->{sink}')
    return list(nx.all_shortest_paths(as_graph, source, sink))


def _candidates_all_paths_starting_shortest(
        as_graph: nx.DiGraph, source: int, sink: int,
        timeout_s: float, max_paths: int
) -> t.List[t.List[int]]:
    """Realistically, this will only compute paths up to length 5"""
    _LOG.info(f'Getting All Paths {source}->{sink}')
    time_start = time()
    ret: t.List[t.List[int]] = []
    iter = 0
    for path in nx.shortest_simple_paths(as_graph, source, sink):
        ret.append(path)

        iter += 1
        if iter % 500 == 0:
            # _LOG.info(f'Found {iter} paths')
            if time() - time_start > timeout_s:
                _LOG.info(
                    f'Path search timed out after {timeout_s}s ' +
                    f' with {iter} paths found'
                )
                break

        if iter >= max_paths:
            _LOG.info(
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
def _get_as_graph() -> nx.DiGraph:
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
    return bgp_graph_to_networkx(as_graph_cache.get())

