import logging
import typing as t

import networkx as nx
from bgpsyche.service.mrt_file_parser import ASPathMeta

_LOG = logging.getLogger(__name__)


ASGraphCounted = t.Dict[int, t.Dict[int, int]]
Dest2ASGraph = t.Dict[int, ASGraphCounted]


def as_graphs_from_paths(
        paths: t.Iterator[ASPathMeta]
) -> t.Tuple[ASGraphCounted, Dest2ASGraph]:
    dest2graph: Dest2ASGraph = {}
    graph_full: ASGraphCounted = {}

    for path in paths:
        dst_asn = path['path'][-1]
        for i in range(len(path['path']) - 1):
            src, dst = path['path'][i], path['path'][i+1]

            # dest2graph: src -> dst
            if dst_asn not in dest2graph: dest2graph[dst_asn] = {}
            if src not in dest2graph[dst_asn]: dest2graph[dst_asn][src] = {}
            if dst not in dest2graph[dst_asn][src]: dest2graph[dst_asn][src][dst] = 0
            dest2graph[dst_asn][src][dst] += path['count']

            # dest2graph: src -> dst
            if dst not in dest2graph[dst_asn]: dest2graph[dst_asn][dst] = {}
            if src not in dest2graph[dst_asn][dst]: dest2graph[dst_asn][dst][src] = 0
            dest2graph[dst_asn][dst][src] += path['count']

            # graph_full: src -> dst
            if src not in graph_full: graph_full[src] = {}
            if dst not in graph_full[src]: graph_full[src][dst] = 0
            graph_full[src][dst] += path['count']

            # graph_full: src -> dst
            if dst not in graph_full: graph_full[dst] = {}
            if src not in graph_full[dst]: graph_full[dst][src] = 0
            graph_full[dst][src] += path['count']

    return graph_full, dest2graph


def bgp_graph_to_networkx(as_graph: ASGraphCounted) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(as_graph.keys())
    for as_source, sink2count in as_graph.items():
        for as_sink in sink2count.keys():
            g.add_edge(as_source, as_sink)
    _LOG.info(f'Got BGP Graph with {g.number_of_nodes()} nodes')
    return g