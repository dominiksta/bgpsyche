from datetime import datetime
import itertools
import logging
import typing as t

import bgpsyche.logging_config
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.ext import ripe_ris, routeviews, mrt_custom
from bgpsyche.service.bgp_graph import EDGE_LINK_COUNT, as_graphs_from_paths

_LOG = logging.getLogger(__name__)
_mkdate = datetime.fromisoformat

# config / data input definitions
# ----------------------------------------------------------------------

def _ris_routeviews_single_rib():
    return itertools.chain(
        ripe_ris.iter_paths(_mkdate('2023-05-01T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-01T00:00')),
    )

def _ris_routeviews_week_of_ribs():
    return itertools.chain(
        ripe_ris.iter_paths(_mkdate('2023-05-01T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-02T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-03T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-04T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-05T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-06T00:00')),
        ripe_ris.iter_paths(_mkdate('2023-05-07T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-01T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-02T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-03T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-04T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-05T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-06T00:00')),
        routeviews.iter_paths(_mkdate('2023-05-07T00:00')),
    )

_INPUT_DATA_FUN = _ris_routeviews_week_of_ribs

# compute
# ----------------------------------------------------------------------

def _get_link_counts(graph: nx.Graph) -> t.List[int]:
    _LOG.info(
        f'Graph has {graph.number_of_edges()} links ' +
        f'(and {graph.number_of_nodes()} ASes)'
    )
    iter = t.cast(t.Iterator[t.Tuple[int, int, int]],
                  graph.edges.data(t.cast(t.Any, EDGE_LINK_COUNT)))
    return [ el[2] for el in iter ]

def _plot_link_counts(link_counts: t.List[int]) -> None:
    percent_n = lambda n: link_counts.count(n) / len(link_counts)

    def cumulative(n: int):
        if n == 1: return percent_n(n)
        else: return percent_n(n) + cumulative(n - 1)

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    axs = t.cast(t.Any, fig.subplots(1, 2))

    def draw_helper_line(plt_i: int, n: int):
        axs[plt_i].plot(
            [0, 10], [cumulative(n), cumulative(n)],
            linestyle='dotted', color='black'
        )

    axs[0].ecdf(link_counts)
    draw_helper_line(0, 1)
    draw_helper_line(0, 2)
    draw_helper_line(0, 3)
    axs[0].set_xlim([0, 10])
    axs[0].set_yticks(np.array([
        0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1,
        round(cumulative(1), 2),
        round(cumulative(2), 2),
        round(cumulative(3), 2),
    ]))
    axs[0].set_xticks(range(11))
    axs[0].set_ylabel('CDF')
    axs[0].set_xlabel('Link Seen Count')

    axs[1].ecdf(link_counts)
    axs[1].set_xlim([0, 100])
    axs[1].set_ylabel('CDF')
    axs[1].set_xticks(range(0, 101, 10))
    axs[1].set_xlabel('Link Seen Count')

    plt.show()

# main
# ----------------------------------------------------------------------

def _research_as_graph_redundancy():
    cache = PickleFileCache(
        'research_as_graph_completeness_' + _INPUT_DATA_FUN.__name__,
        lambda: as_graphs_from_paths(_INPUT_DATA_FUN())[0]
    )
    # cache.invalidate()
    graph = cache.get()
    _plot_link_counts(_get_link_counts(graph))



if __name__ == '__main__': _research_as_graph_redundancy()