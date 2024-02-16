import typing as t
import logging
from time import time

import networkx as nx
from bgpsyche.stage1_candidates.types import WeightFunction

_LOG = logging.getLogger(__name__)

class PathCandidatesRes(t.TypedDict):
    candidates: t.List[t.List[int]]
    by_length: t.Dict[int, t.List[t.List[int]]]

PathCandidatesByLen = t.Dict[int, t.List[t.List[int]]]


def abort_on_timeout(timeout_s: float) -> t.Callable[[t.List[int]], bool]:
    before = time()
    return lambda _: (time() - before) > timeout_s

def abort_on_amount(amount: float) -> t.Callable[[t.List[int]], bool]:
    found = 0
    def abort_on_amount_inner(_):
        nonlocal found
        found += 1
        return found >= amount
    return abort_on_amount_inner

class GetPathCandidatesAbortCondition(t.TypedDict):
    func: t.Callable[[t.List[int]], bool]
    desc: str

GetPathCandidatesAbortConditions = \
    t.Callable[[], t.List[GetPathCandidatesAbortCondition]]

def get_path_candidates_from_graph(
        as_graph: nx.Graph,
        source: int, sink: int,
        abort_on: GetPathCandidatesAbortConditions,
        quiet: bool = False,
        weight: t.Optional[WeightFunction] = None,
) -> PathCandidatesRes:
    if not quiet: _LOG.info(f'Getting Path Candidates {source}->{sink}')
    _abort_on = abort_on()

    paths_iter = nx.shortest_simple_paths(as_graph, source, sink, weight=weight)

    candidates: t.List[t.List[int]] = []
    iter = 0
    abort = False
    try:
        for path in paths_iter:
            if abort: break

            candidates.append(path)
            iter += 1

            for abort_condition in _abort_on:
                if abort_condition['func'](path):
                    if not quiet: _LOG.info(
                            f'Path search {source}->{sink} aborted ' +
                            f'after {iter} found: ' +
                            f' {abort_condition["desc"]}'
                    )
                    abort = True
    except nx.NetworkXNoPath:
        _LOG.info(f'No path in graph between {source} and {sink}')


    return {
        'candidates': candidates,
        'by_length': _mk_paths_dict_by_length(candidates),
    }


def _mk_paths_dict_by_length(
        paths: t.List[t.List[int]]
) -> t.Dict[int, t.List[t.List[int]]]:
    ret: t.Dict[int, t.List[t.List[int]]] = {}
    for path in paths:
        l = len(path)
        if l not in ret: ret[l] = []
        ret[l].append(path)
    return ret
