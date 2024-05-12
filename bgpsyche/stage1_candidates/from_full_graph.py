import typing as t
from datetime import datetime
import functools
import itertools

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.service.bgp_graph import as_graphs_from_paths
from bgpsyche.service.ext import ripe_ris, routeviews
from bgpsyche.stage1_candidates.from_graph import (
    GetPathCandidatesAbortCondition, PathCandidatesRes, abort_on_amount,
    abort_on_timeout, get_path_candidates_from_graph
)
from bgpsyche.stage1_candidates.types import WeightFunction


def get_path_candidates_full_graph(
        source: int, sink: int,
        abort_on: t.Callable[
            [], t.List[GetPathCandidatesAbortCondition]
        ] = lambda: [
            { 'func': abort_on_timeout(3), 'desc': 'timeout 5s' },
            { 'func': abort_on_amount(800), 'desc': 'amount 4k' },
        ],
        quiet: bool = False,
        weight: t.Optional[WeightFunction] = None,
) -> PathCandidatesRes:
    return get_path_candidates_from_graph(
        as_graph_full(),
        source, sink, abort_on, quiet, weight
    )



@functools.lru_cache()
def as_graph_full(dt = datetime.fromisoformat('2023-05-01T00:00')):
    cache = PickleFileCache(
        f'as_graph_{dt.strftime("%Y%m%d")}',
        lambda: as_graphs_from_paths(itertools.chain(
            ripe_ris.iter_paths(dt),
            routeviews.iter_paths(dt),
        ))[0]
    )

    # cache.invalidate()
    return cache.get()
