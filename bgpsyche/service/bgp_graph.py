import typing as t
from datetime import datetime
import itertools
import functools
import logging

import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.mrt_file_parser import ASPathMeta
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.util.run_in_pypy import run_in_pypy

_LOG = logging.getLogger(__name__)

Dest2ASGraph = t.Dict[int, nx.Graph]

EDGE_LINK_COUNT = 'edge_link_count'

def as_graphs_from_paths(
        paths: t.Iterator[ASPathMeta]
) -> t.Tuple[nx.Graph, Dest2ASGraph]:
    dest2graph: Dest2ASGraph = {}
    graph_full = nx.Graph()

    for path in paths:
        dst_asn = path['path'][-1]
        for i in range(len(path['path']) - 1):
            src, dst = path['path'][i], path['path'][i+1]

            # dest2graph
            if dst_asn not in dest2graph: dest2graph[dst_asn] = nx.Graph()
            dest2graph[dst_asn].add_edge(src, dst)
            if EDGE_LINK_COUNT not in dest2graph[dst_asn][src][dst]:
                dest2graph[dst_asn][src][dst][EDGE_LINK_COUNT] = 0
            dest2graph[dst_asn][src][dst][EDGE_LINK_COUNT] += 1

            # graph_full
            graph_full.add_edge(src, dst)
            if EDGE_LINK_COUNT not in graph_full[src][dst]:
                graph_full[src][dst][EDGE_LINK_COUNT] = 0
            graph_full[src][dst][EDGE_LINK_COUNT] += 1


    _LOG.info(f'Got BGP Graph with {graph_full.number_of_nodes()} nodes')
    return graph_full, dest2graph


@functools.lru_cache()
@run_in_pypy(cache=PickleFileCache)
def as_graph_from_ext(
        routeviews_dts = [
            datetime.fromisoformat('2023-05-01T00:00')
        ],
        ripe_ris_dts = [
            datetime.fromisoformat('2023-05-01T00:00')
        ]
) -> nx.Graph:
    return as_graphs_from_paths(itertools.chain(
        *[ripe_ris.iter_paths(dt) for dt in ripe_ris_dts],
        *[routeviews.iter_paths(dt) for dt in routeviews_dts],
    ))[0]
