from datetime import datetime
import functools
import itertools
import typing as t

from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.util.net.prefix_tree import PrefixTree
from bgpsyche.service.ext import routeviews, ripe_ris

def _subpaths(path: t.List[int]) -> t.List[t.List[int]]:
    return [
        path[i:j]
        for i in range(len(path))
        for j in range(i + 1, len(path) + 1)
        if len(path[i:j]) >= 2
    ]



_ris_for_date = \
    lambda dt: ripe_ris.iter_paths(
        datetime.fromisoformat(dt), eliminate_path_prepending=True
    )
_routeviews_for_date = \
    lambda dt: routeviews.iter_paths(
        datetime.fromisoformat(dt), eliminate_path_prepending=True
    )


@functools.lru_cache()
def _bgp_prefix_tree():
    cache = PickleFileCache(
        'bgp_prefix_tree_snippet_length',
        lambda: PrefixTree.from_iter(
            path_meta['path'] for path_meta in (itertools.chain(
                _ris_for_date('2023-05-01T00:00'),
                _routeviews_for_date('2023-05-01T00:00'),
            ))
        )
    )
    # cache.invalidate()
    return cache.get()


def longest_real_snippet(path: t.List[int]) -> t.List[int]:
    max_path = []
    tree = _bgp_prefix_tree()

    for subpath in _subpaths(path):
        s = tree.search(subpath)
        if s and len(subpath) > len(max_path): max_path = subpath

    return max_path
