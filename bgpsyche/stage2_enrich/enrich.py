import typing as t

from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.asrank import get_asrank
from bgpsyche.service.ext.ripe_as_names_countries import (
    get_ripe_as_country, get_ripe_as_name
)
from bgpsyche.stage2_enrich.types import ASFeaturesRaw

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


def enrich_path(path: t.List[int]) -> t.List[ASFeaturesRaw]:
    return [ enrich_asn(asn) for asn in path ]