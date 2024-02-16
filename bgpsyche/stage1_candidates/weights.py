from collections import defaultdict
import typing as t
from datetime import datetime
from pprint import pformat

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.stage1_candidates.from_full_graph import as_graph_full
from bgpsyche.stage1_candidates.types import WeightFunction
from bgpsyche.util.bgp.relationship import RelationshipKind, Source2Sink2Rel

def weight_by_relationship(sink: int) -> WeightFunction:
    dt = datetime.fromisoformat('2023-05-01T00:00')
    weights = _as_graph_weights_for_sink(
        sink,
        get_caida_asrel(dt),
        as_graph_full(),
    )
    def _weight(source: int, sink: int, _: t.Dict[str, t.Any]) -> float:
        assert source in weights, pformat({'source': source})
        assert sink in weights[source], pformat({
            'source': source, 'sink': sink, 'weights[source]': weights[source]
        })
        return weights[source][sink]
    return _weight


def _as_graph_weights_for_sink(
        sink: int,
        rels: Source2Sink2Rel,
        full_graph: nx.Graph,
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

        _, asn2dist = nx.dijkstra_predecessor_and_distance(full_graph, sink, cutoff=12)
        asn2dist = t.cast(t.Dict[int, int], asn2dist)

        ret: t.Dict[int, t.Dict[int, float]] = defaultdict(dict)

        for src, dst in full_graph.edges():
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
