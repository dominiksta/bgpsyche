import typing as t
import itertools
import logging
from time import time

import networkx as nx
from bgpsyche.service.bgp_graph import as_graph_from_ext

_LOG = logging.getLogger(__name__)

class PathCandidatesRes(t.TypedDict):
    candidates: t.List[t.List[int]]
    by_length: t.Dict[int, t.List[t.List[int]]]


PathCandidatesByLen = t.Dict[int, t.List[t.List[int]]]


def abort_on_timeout(timeout_s: float) -> t.Callable[[t.List[int]], bool]:
    before = time()
    return lambda _path: (time() - before) > timeout_s

def abort_on_amount(amount: float) -> t.Callable[[t.List[int]], bool]:
    found = 0
    def abort_on_amount_inner(_):
        nonlocal found
        found += 1
        return found >= amount
    return abort_on_amount_inner

class _GetPathCandidatesAbortConditions(t.TypedDict):
    func: t.Callable[[t.List[int]], bool]
    desc: str


def get_path_candidates(
        source: int, sink: int,
        abort_on: t.Union[
            t.Literal['default'],
            t.List[_GetPathCandidatesAbortConditions]
        ] = 'default',
        quiet: bool = False,
        unordered: bool = False,
) -> PathCandidatesRes:
    if not quiet: _LOG.info(f'Getting Path Candidates {source}->{sink}')
    as_graph  = t.cast(nx.Graph, as_graph_from_ext())
    if abort_on == 'default': abort_on = [
            { 'func': abort_on_timeout(5), 'desc': 'timeout 5s' },
            { 'func': abort_on_amount(4000), 'desc': 'amount 4k' },
    ]

    paths_iter = {
        True: nx.all_simple_paths(as_graph, source, sink, cutoff=8),
        False: nx.shortest_simple_paths(as_graph, source, sink),
    }[unordered]

    candidates: t.List[t.List[int]] = []
    iter = 0
    abort = False
    for path in paths_iter:
        if abort: break

        candidates.append(path)
        iter += 1

        for abort_condition in abort_on:
            if abort_condition['func'](path):
                if not quiet: _LOG.info(
                        f'Path search {source}->{sink} aborted ' +
                        f'after {iter} found: ' +
                        f' {abort_condition["desc"]}'
                )
                abort = True


    return {
        'candidates': candidates,
        'by_length': _mk_paths_dict_by_length(candidates),
    }


def flatten_candidates(
        candidates: PathCandidatesByLen
) -> t.List[t.List[int]]:
    return list(itertools.chain.from_iterable(candidates.values()))


def _mk_paths_dict_by_length(
        paths: t.List[t.List[int]]
) -> t.Dict[int, t.List[t.List[int]]]:
    ret: t.Dict[int, t.List[t.List[int]]] = {}
    for path in paths:
        l = len(path)
        if l not in ret: ret[l] = []
        ret[l].append(path)
    return ret
