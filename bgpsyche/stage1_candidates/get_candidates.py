from collections import defaultdict
from datetime import datetime
import functools
from pprint import pformat
import typing as t
import itertools
import logging
from time import time

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.util.bgp.relationship import RelationshipKind
from bgpsyche.service.ext import ripe_ris, routeviews

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

class _GetPathCandidatesAbortConditions(t.TypedDict):
    func: t.Callable[[t.List[int]], bool]
    desc: str

WeightFunction = t.Callable[[int, int, t.Dict[str, t.Any]], float]

def get_path_candidates(
        source: int, sink: int,
        abort_on: t.Callable[
            [], t.List[_GetPathCandidatesAbortConditions]
        ] = lambda: [
            { 'func': abort_on_timeout(5), 'desc': 'timeout 5s' },
            { 'func': abort_on_amount(4000), 'desc': 'amount 4k' },
        ],
        quiet: bool = False,
        weight: t.Optional[WeightFunction] = None,
) -> PathCandidatesRes:
    if not quiet: _LOG.info(f'Getting Path Candidates {source}->{sink}')
    as_graph = t.cast(nx.Graph, _as_graph_main())
    _abort_on = abort_on()

    paths_iter = nx.shortest_simple_paths(as_graph, source, sink, weight=weight)

    candidates: t.List[t.List[int]] = []
    iter = 0
    abort = False
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


@functools.lru_cache()
def _as_graph_main(dt = datetime.fromisoformat('2023-05-01T00:00')):
    cache = PickleFileCache(
        f'as_graph_{dt.strftime("%Y%m%d")}',
        lambda: as_graphs_from_paths(itertools.chain(
            ripe_ris.iter_paths(dt),
            routeviews.iter_paths(dt),
        ))[0]
    )

    # cache.invalidate()
    return cache.get()


def weight_by_relationship(sink: int) -> WeightFunction:
    dt = datetime.fromisoformat('2023-05-01T00:00')
    weights = _as_graph_weights_for_sink(sink, dt)
    def _weight(source: int, sink: int, _: t.Dict[str, t.Any]) -> float:
        assert source in weights, pformat({'source': source})
        assert sink in weights[source], pformat({
            'source': source, 'sink': sink, 'weights[source]': weights[source]
        })
        return weights[source][sink]
    return _weight


def _as_graph_weights_for_sink(
        sink: int,
        dt = datetime.fromisoformat('2023-05-01T00:00'),
) -> t.Dict[int, t.Dict[int, float]]:

    # determined empirically from routeviews and ris (2023-05-01T00:00)
    weight_mapping_by_path_pos: t.Dict[int, t.Dict[t.Optional[RelationshipKind], float]] = {
         0: { 'c2p': 0.76, 'p2c': 0.01, 'p2p': 0.23, None: 0.08  },
         1: { 'c2p': 0.11, 'p2c': 0.68, 'p2p': 0.21, None: 0.05  },
         2: { 'c2p': 0.04, 'p2c': 0.93, 'p2p': 0.03, None: 0.03  },
         3: { 'c2p': 0.01, 'p2c': 0.96, 'p2p': 0.03, None: 0.08  },
         4: { 'c2p': 0.01, 'p2c': 0.96, 'p2p': 0.03, None: 0.07  },
         5: { 'c2p': 0.01, 'p2c': 0.95, 'p2p': 0.04, None: 0.02  },
         6: { 'c2p': 0.01, 'p2c': 0.81, 'p2p': 0.18, None: 0.02  },
         7: { 'c2p': 0.01, 'p2c': 0.78, 'p2p': 0.21, None: 0.02  },
         8: { 'c2p': 0.02, 'p2c': 0.79, 'p2p': 0.20, None: 0.01  },
         9: { 'c2p': 0.01, 'p2c': 0.74, 'p2p': 0.25, None: 0.00  },
        10: { 'c2p': 0.00, 'p2c': 0.83, 'p2p': 0.17, None: 0.00  },
        11: { 'c2p': 0.00, 'p2c': 0.55, 'p2p': 0.45, None: 0.00  },
        12: { 'c2p': 0.00, 'p2c': 1.00, 'p2p': 0.00, None: 0.00  },
    }

    def _get():
        rels = get_caida_asrel(dt)
        g_main = _as_graph_main(dt)

        _, asn2dist = nx.dijkstra_predecessor_and_distance(g_main, sink, cutoff=12)
        asn2dist = t.cast(t.Dict[int, int], asn2dist)

        ret: t.Dict[int, t.Dict[int, float]] = defaultdict(dict)

        for src, dst in g_main.edges():
            rel = rels[src][dst] if src in rels and dst in rels[src] else None
            weight = weight_mapping_by_path_pos[
                min(asn2dist[src], len(weight_mapping_by_path_pos) - 1)
            ][rel]
            ret[src][dst] = weight
            ret[dst][src] = weight

        return dict(ret)

    cache = PickleFileCache(
        f'as_graph_weights_for_sink_{sink}',
        _get
    )

    # cache.invalidate()
    return cache.get()



# def _test():
#     source, sink = 23673, 24371

#     g: t.Any = _as_graph_main()

#     # print(nx.bfs_tree(g, source))
#     source2paths: t.Dict[int, t.List[t.List[int]]] = defaultdict(list)
#     visited: t.Set[int] = set()

#     iter = 0
#     for current_node, _ in nx.bfs_edges(g, sink):
#         if current_node in visited: continue
#         visited.add(current_node)
#         iter += 1
#         if iter % 100 == 0:
#             print(iter)
#             print(pformat(source2paths))

#         for neighbour in g[current_node]:
#             if neighbour == sink: source2paths[current_node].append([sink])
#             if neighbour in source2paths:
#                 for path in source2paths[neighbour]:
#                     # print(path)
#                     source2paths[current_node].append([neighbour] + path)

#         # if dst == sink: source2paths[src].append([dst])
#         # if dst in source2paths:
#         #     for path in source2paths[dst]:
#         #         source2paths[src].append([dst] + path)


#     print(pformat(source2paths))


        



# if __name__ == '__main__': _test()