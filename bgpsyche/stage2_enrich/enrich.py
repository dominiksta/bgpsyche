from datetime import datetime
import statistics
import typing as t

from bgpsyche.service.bgp_path_snippet_length import longest_real_snippet
from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.asrank import get_asrank
from bgpsyche.service.ext.caida_asrel import get_caida_asrel
from bgpsyche.service.ext.ripe_as_names_countries import (
    get_ripe_as_country, get_ripe_as_name
)
from bgpsyche.stage2_enrich.geographic_distance import geographic_distance_diff
from bgpsyche.util.bgp.valley_free import path_is_valley_free
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, PathFeatures

def _maybe(d: t.Any, key: str) -> t.Any: return d[key] if d else None


def enrich_asn(asn: int) -> ASFeaturesRaw:
    pdb_info = peeringdb.Client.get_network_by_asn(asn)

    return {
        'asn': asn,
        'ripe_name': get_ripe_as_name(asn),
        'ripe_country': get_ripe_as_country(asn),
        'as_rank': get_asrank(asn),
        'peeringdb_type': _maybe(pdb_info, 'info_type'),
        'peeringdb_prefix_count_v4': _maybe(pdb_info, 'info_prefixes4'),
        'peeringdb_prefix_count_v6': _maybe(pdb_info, 'info_prefixes6'),
        'peeringdb_geographic_scope': _maybe(pdb_info, 'info_scope'),
    }


def enrich_path(path: t.List[int]) -> PathFeatures:
    dt = datetime.fromisoformat('2023-05-01T00:00')

    return {
        'length': len(path),
        'is_valley_free': path_is_valley_free(get_caida_asrel(dt), path),
        'geographic_distance_diff': geographic_distance_diff(path)[0],
        'asrank_variance': statistics.variance([get_asrank(asn) for asn in path]),
        'longest_real_snippet_diff': len(path) - len(longest_real_snippet(path)),
    }