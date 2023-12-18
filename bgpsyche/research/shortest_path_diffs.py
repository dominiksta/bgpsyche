import typing as t
from datetime import datetime
import statistics

import bgpsyche.logging_config
import editdistance
import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext import ripe_ris, routeviews


def _research_compute_shortest_paths_diffs():
    as_graph_cache = PickleFileCache(
        'as_graphs',
        lambda: as_graphs_from_paths(ripe_ris.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00')
        ))[0]
    )

    # as_graph_cache.invalidate()
    as_graph = as_graph_cache.get()

    edit_distances: t.List[float] = []
    exact_matches = 0
    missing_node = 0
    iter = 0
    for path_meta in routeviews.iter_paths(
            datetime.fromisoformat('2023-05-01T00:00')
    ):
        iter += 1
        p = path_meta['path']
        try:
            shortest_paths = nx.all_shortest_paths(as_graph, p[0], p[-1])
            best_distance = min([ editdistance.eval(p, sp) for sp in shortest_paths ])
            if best_distance == 0: exact_matches += 1
            edit_distances.append(best_distance)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            missing_node += 1
        if iter % 100 == 0:
            print(f'iter      : {iter}')
            print(f'dist_mean : {statistics.mean(edit_distances)}')
            print(f'exact     : {exact_matches / iter}')
            print(f'missing   : {missing_node / iter}')


if __name__ == '__main__': _research_compute_shortest_paths_diffs()