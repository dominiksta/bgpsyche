import typing as t

from bgpsyche.service.ext import peeringdb
from bgpsyche.util.geo import CountryName

class ASFeatures(t.TypedDict):
    asn: int
    ripe_name: str
    ripe_country: CountryName
    as_rank: int
    peeringdb_type: t.Optional[peeringdb.NetworkType]
    peeringdb_prefix_count_v4: t.Optional[int]
    peeringdb_prefix_count_v6: t.Optional[int]
    peeringdb_geographic_scope: t.Optional[peeringdb.NetworkScope]



# class EnrichedPath(t)