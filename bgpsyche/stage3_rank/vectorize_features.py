from statistics import mean
import typing as t
from time import mktime
from bgpsyche.service.ext.asrank import ASRANK_CUSTOMER_CONE_SIZE_RANGE

from bgpsyche.stage2_enrich.as_category import AS_CATEGORY
from bgpsyche.stage2_enrich.democracy_index import democracy_index_avg
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, LinkFeatures, PathFeatures
from bgpsyche.stage3_rank.vectorize_util import one_hot

PATH_FEATURE_VECTOR_NAMES: t.List[str] = [
    'length',
    'is_valley_free',
]

def vectorize_path_features(features: PathFeatures) -> t.List[t.Union[int, float]]:
    return [
        features['length'],
        0 if features['is_valley_free'] is None
        else int(features['is_valley_free']),
    ]

LINK_FEATURE_VECTOR_NAMES: t.List[str] = [
    'rel_p2c',
    'rel_p2p',
    'rel_c2p',
    'rel_unknown',
    # 'distance_km',
    # 'trade_service_volume_million_usd',
]

def vectorize_link_features(features: LinkFeatures) -> t.List[t.Union[int, float]]:
    return [
        int(features['rel'] == 'p2c'),
        int(features['rel'] == 'p2p'),
        int(features['rel'] == 'c2p'),
        int(features['rel'] is None),
        # features['distance_km'],
        # features['trade_service_volume_million_usd'],
    ]

AS_FEATURE_VECTOR_NAMES: t.List[str] = [
    # 'country_democracy_index',
    'as_rank_cone',
    'rirstat_born',
    'rirstat_addr_count_v4',
    'rirstat_addr_count_v6',
    'category_unknown',
    'category_transit_access',
    'category_content',
    'category_enterprise',
    'category_educational_research',
    'category_non_profit',
    'category_route_server',
    'category_network_services',
    'category_route_collector',
    'category_government',
]

_one_hot_as_category = one_hot(list(AS_CATEGORY), optional=False)

def vectorize_as_features(features: ASFeaturesRaw) -> t.List[t.Union[int, float]]:
    as_rank_cone_scaled = (
        0 if features['as_rank_cone'] is None else \
        features['as_rank_cone'] /
        (ASRANK_CUSTOMER_CONE_SIZE_RANGE[1] - ASRANK_CUSTOMER_CONE_SIZE_RANGE[0])
    )

    return [
        # features['country_democracy_index'] or democracy_index_avg(),
        as_rank_cone_scaled,
        mktime(features['rirstat_born'].timetuple())
        if features['rirstat_born'] is not None else -1,
        features['rirstat_addr_count_v4'] or -1,
        features['rirstat_addr_count_v6'] or -1,
        *_one_hot_as_category(features['as_category']),
    ]