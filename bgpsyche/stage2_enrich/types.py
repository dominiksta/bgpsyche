import typing as t

from bgpsyche.service.ext import peeringdb
from bgpsyche.service.ext.ripe_as_names_countries import AsnTxtCountryCode

class ASFeaturesRaw(t.TypedDict):
    asn: int
    ripe_name: str
    ripe_country: AsnTxtCountryCode
    as_rank: int
    peeringdb_type: t.Optional[peeringdb.NetworkType]
    peeringdb_prefix_count_v4: t.Optional[int]
    peeringdb_prefix_count_v6: t.Optional[int]
    peeringdb_geographic_scope: t.Optional[peeringdb.NetworkScope]



class PathFeatures(t.TypedDict):
    length: int

    # Intuition: A path that goes from a source in Germany to a destination in
    # Germany is likely not going to contain an AS in China.
    #
    # Computation: This value is computed by adding the geographic distance
    # between the AS hops and then subtracting the distance between the source
    # and destination AS. Theoretically, a larger value should mean that the
    # path is less likely. Since a single AS may have multiple prefixes, we
    # compute all possible distances.
    #
    # Dealing with unlocatable hops: We have shown that around 99.75% of all BGP
    # routed prefixes are geolocatable using MaxMind GeoIP2 Lite (which we use
    # here). We ignore ASes that
    geographic_distance_diff: t.List[float]