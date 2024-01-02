from functools import reduce
import logging
from pprint import pformat
import typing as t
import operator

import networkx as nx
from bgpsyche.service.ext.maxmind_geoip2_lite import geoip2_find_network
from bgpsyche.service.ext.prefix2as import as2prefix
from bgpsyche.util.geo import COUNTRY_DISTANCES, Alpha2WithLocation

_T = t.TypeVar('_T')
_LOG = logging.getLogger(__name__)

# determined by experiment
_GEOGRAPHIC_DISTANCE_DIFF_AVG = 175

GeographicDistanceDiffRetMeta = t.Literal[
    'src_or_dest_unlocatable',
    'paths_max_len'
]

def geographic_distance_diff(path: t.List[int]) -> t.Tuple[
        float, t.Set[GeographicDistanceDiffRetMeta]
]:
    """See `geographic_distance_diff` in `PathFeatures`"""

    prefix_sets = [ as2prefix(asn) for asn in path ]
    alpha2: t.List[t.Set[Alpha2WithLocation]] = [
        {
            p for p in
            set(geoip2_find_network(prefix) for prefix in prefix_set)
            if p is not None
        }
        for prefix_set in prefix_sets
    ]

    if len(alpha2[0]) == 0 or len(alpha2[-1]) == 0:
        return _GEOGRAPHIC_DISTANCE_DIFF_AVG, { 'src_or_dest_unlocatable' }

    distances_source_dest = [
        COUNTRY_DISTANCES[src][dst]
        for dst in alpha2[-1]
        for src in alpha2[0]
    ]

    paths, paths_max_len = t.cast(
        t.Tuple[t.List[t.List[Alpha2WithLocation]], bool], _path_alternatives(alpha2)
    )

    distances_paths = [
        sum([COUNTRY_DISTANCES[p[i]][p[i+1]] for i in range(len(p) - 1)])
        for p in paths
    ]

    return min([
        abs(distance_source_dest - distance_path)
        for distance_source_dest in distances_source_dest
        for distance_path in distances_paths
    ]), { 'paths_max_len' } if paths_max_len else set()


def _path_alternatives(
        it: t.Iterable[t.Iterable[_T]],
        max_out = 50_000
) -> t.Tuple[t.List[t.List[_T]], bool]:

    it_l = [ list(inner) for inner in it ]
    it_l = [ el for el in it_l if len(el) != 0 ]

    g = nx.DiGraph()
    for i in range(len(it_l) - 1):
        inner_curr_l = it_l[i]
        inner_next_l = it_l[i+1]
        for inner_curr in inner_curr_l:
            for inner_next in inner_next_l:
                g.add_edge((i, inner_curr), (i + 1, inner_next))

    found, stop = [], False

    for src in it_l[0]:
        for dst in it_l[-1]:
            try:
                for p in nx.all_simple_paths(
                    g, (0, src), (len(it_l) - 1, dst)
                ):
                    if len(found) >= max_out:
                        stop = True
                        break
                    found.append(p)
            except:
                _LOG.error(pformat({
                    'it': it_l,
                    'edges': list(g.edges),
                    'nodes': list(g.nodes),
                    'found_so_far_max_100': found[:100],
                }))
                raise

        if stop:
            _LOG.warning(
                f'Found more then {max_out} path alternatives ({len(found)}), ' +
                f'ignoring any further paths'
            )
            # print(len(found), it_l)
            break

    out: t.List[t.List[_T]] = [
        [ tup[1] for tup in path ]
        for path in found
    ]

    if __debug__:
        len_should = \
            min(max_out, reduce(operator.mul, [ len(inner) for inner in it_l ]))
        assert len(out) == len_should, pformat({
            'len_out': len(out), 'len_should': len_should,
            'it': it_l, 'out_max_100': out[:100],
        })

    return out, len(out) >= max_out