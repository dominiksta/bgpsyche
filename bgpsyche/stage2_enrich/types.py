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
    is_valley_free: t.Optional[bool]
    # Intuition: A path that goes from a source in Germany to a destination in
    # Germany is likely not going to contain an AS in China.
    #
    # Computation: This value is computed by adding the geographic distance
    # between the AS hops and then subtracting the distance between the source
    # and destination AS. Theoretically, a larger value should mean that the
    # path is less likely. Since a single AS may have multiple prefixes, we
    # compute all possible distances.
    #
    # NOTE: Dealing with ASes that announce multiple locatable prefixes: We
    # compute all possible values for `geographic_distance_diff` and then choose
    # the *smallest*. (This could possibly be improved by comparing against an
    # AS graph constructed from traceroutes. Doing so would require a lot of
    # effort though, since getting a large enough part of the AS graph using
    # traceroutes is nowhere near as easy as using RouteViews/RIS data.)
    # Additionally, there are many AS paths with a *really* large set of
    # possible prefix level paths. We only consider the first 50k prefix level
    # paths computed to avoid freezing the program.
    #
    # Dealing with unlocatable hops: We have shown that around 99.75% of all BGP
    # routed prefixes are geolocatable using MaxMind GeoIP2 Lite (which we use
    # here). For the few unlocatable ASes we use the average position of their
    # immediate peers. In the very unlikely event that all of them are
    # unlocatable as well, we do one of two things depending on the position of
    # the hop in the path:
    # - If the unlocatable hop is in the middle of the path, we ignore it in the
    #   computation entirely.
    # - If the unlocatable hop is the source or destination AS, we will use the
    #   average final value of `geographic_distance_diff` for all paths, which
    #   is 175km.
    # geographic_distance_diff: float
    # # The amount of peaks in the asrank curve.
    # asrank_peaks: int
    asrank_variance: float
    # The difference between the length of the path and the longest subpath seen
    # in real paths. Larger values should indicate a less likely path.
    longest_real_snippet_diff: int