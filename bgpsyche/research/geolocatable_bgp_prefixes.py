from datetime import datetime
import functools
import itertools
import logging
import typing as t
from bgpsyche.logging_config import logging_setup

from bgpsyche.service.ext import ripe_ris
from bgpsyche.service.ext.maxmind_geoip2_lite import geoip2_find_network
from bgpsyche.util.benchmark import bench_function
from bgpsyche.util.net.prefix_tree import (
    NetworkPrefixTreeV4, NetworkPrefixTreeV6, make_prefix_trees_for_list
)

logging_setup()
_LOG = logging.getLogger(__name__)

@functools.lru_cache
@bench_function
def _get_bgp_routed_prefixes() -> t.Tuple[NetworkPrefixTreeV4, NetworkPrefixTreeV6]:
    return make_prefix_trees_for_list((
        meta['dst_prefix'] for meta in ripe_ris.iter_paths_with_prefix(
            datetime.fromisoformat('2023-05-01T00:00')
        )
    ))

def _research_geolocatable_bgp_prefixes():
    iter = 0
    is_eu = 0
    found = 0

    tree_v4, tree_v6 = _get_bgp_routed_prefixes()

    _LOG.info(f'BGP routes v4 prefixes: {len(tree_v4.networks)}')
    _LOG.info(f'BGP routes v6 prefixes: {len(tree_v6.networks)}')

    for net in itertools.chain(tree_v4.networks, tree_v6.networks):
        iter += 1
        if iter % 10_000 == 0: _LOG.info(f'Geolocated prefixes: {found}/{iter}')
        current_found = geoip2_find_network(net)
        if current_found is not None:
            found += 1
            if current_found == 'EU': is_eu += 1

    percent = round((found / iter) * 100, 2)
    _LOG.warning(f'Result: {percent}% ({found}/{iter}, "country" is EU: {is_eu})')


if __name__ == '__main__': _research_geolocatable_bgp_prefixes()