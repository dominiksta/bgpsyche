import logging
import typing as t
from datetime import datetime
import itertools

import networkx as nx
import bgpsyche.logging_config
from bgpsyche.service.bgp_graph import as_graphs_from_paths, bgp_graph_to_networkx
from bgpsyche.service.ext import ripe_ris, routeviews, mrt_custom
from bgpsyche.service.mrt_file_parser import ASPathMeta

_LOG = logging.getLogger(__name__)

# helpers
# ----------------------------------------------------------------------

_MkPathIter = t.Callable[[], t.Iterator[ASPathMeta]]

_mkdate = datetime.fromisoformat

def _get_graph_completeness(
        graph: nx.DiGraph, paths: t.Iterator[ASPathMeta]
) -> t.Tuple[float, int, int]:
    found = 0
    iter = 0

    for path_meta in paths:
        path = path_meta['path']

        iter += 1
        if nx.is_path(graph, path): found += 1

        if iter % 100_000 == 0:
            percent = round((found / iter) * 100, 2)
            _LOG.info(f'Found {percent} ({found}/{iter}) paths in graph')


    percent = round((found / iter) * 100, 2)
    _LOG.info(f'Done: Found {percent} ({found}/{iter}) paths in graph')
    return percent, found, iter


def _get_graph_completeness_both_directions(
        a: _MkPathIter, b: _MkPathIter
):
    a2b = _get_graph_completeness(_mk_graph(a()), b())
    b2a = _get_graph_completeness(_mk_graph(b()), a())
    _LOG.info(f'a2b: *{a2b[0]}%* ({a2b[1]:_}/{a2b[2]:_})')
    _LOG.info(f'b2a: *{b2a[0]}%* ({b2a[1]:_}/{b2a[2]:_})')


def _mk_graph(paths: t.Iterator[ASPathMeta]) -> nx.DiGraph:
    return bgp_graph_to_networkx(as_graphs_from_paths(paths)[0])


# experiments
# ----------------------------------------------------------------------

def _single_day_routeviews_vs_ris(dt: datetime):
    _get_graph_completeness_both_directions(
        lambda: routeviews.iter_paths(dt),
        lambda: ripe_ris.iter_paths(dt),
    )


def _a_week_of_ris_vs_a_day_of_routeviews():
    _get_graph_completeness_both_directions(
        lambda: itertools.chain(
            ripe_ris.iter_paths(_mkdate('2023-05-01T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-02T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-03T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-04T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-05T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-06T00:00')),
            ripe_ris.iter_paths(_mkdate('2023-05-07T00:00')),
        ),
        lambda: routeviews.iter_paths(_mkdate('2023-05-01T00:00')),
    )


def _single_day_tier1_and_ixp_vs_ris_and_routeviews():
    dt = _mkdate('2023-05-01T00:00')
    _get_graph_completeness_both_directions(
        lambda: itertools.chain(
            ripe_ris.iter_paths(dt),
            routeviews.iter_paths(dt),
        ),
        lambda: itertools.chain(
            mrt_custom.iter_paths('mrt_ixp'),
            mrt_custom.iter_paths('mrt_tier1'),
        ),
    )


# main
# ----------------------------------------------------------------------

def _research_as_graph_completeness():
    # _single_day_routeviews_vs_ris(_mkdate('2023-05-01T00:00'))
    _single_day_tier1_and_ixp_vs_ris_and_routeviews()


if __name__ == '__main__': _research_as_graph_completeness()