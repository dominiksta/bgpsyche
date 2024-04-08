from datetime import date
import typing as t

from bgpsyche.stage2_enrich.as_category import ASCategory
from bgpsyche.util.bgp.relationship import RelationshipKind


class LinkFeatures(t.TypedDict):
    rel: t.Optional[RelationshipKind]
    distance_km: int
    trade_service_volume_million_usd: float


class ASFeaturesRaw(t.TypedDict):
    # - censorship index
    # - https://en.wikipedia.org/wiki/List_of_countries_by_number_of_Internet_users
    country_democracy_index: t.Optional[float]

    # TODO: try replacing with our own customer cone computation from t1 mrt data
    as_rank_cone: int
    rirstat_born: t.Optional[date]

    as_category: ASCategory

    # - a smaller value could mean both that the AS is small or that is is a
    #   transit provider. content networks prob have larger prefixes because
    #   they operate more infra.
    # - TODO: try adding feature: how often did something change in rirstat?
    # - alternative: get from bgp routing tables
    #   -> TODO: try replacing with how many & how large prefixes from bgp
    rirstat_addr_count_v4: t.Optional[float]
    rirstat_addr_count_v6: t.Optional[float]


class PathFeatures(t.TypedDict):
    # Both length and is_valley_free may or may not be redundant. The model
    # could learn these properties on an AS level and giving them here
    # additionally could either be helpful or counter productive, we will have
    # to see.
    length: int
    is_valley_free: t.Optional[bool]

    longest_real_snippet: int

    per_dest_markov_confidence: float