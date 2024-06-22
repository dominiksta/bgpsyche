from datetime import datetime
import math
import typing as t
from time import mktime

from bgpsyche.service.ext.asrank import ASRANK_CUSTOMER_CONE_SIZE_RANGE
from bgpsyche.stage2_enrich.as_category import AS_CATEGORY
from bgpsyche.stage2_enrich.democracy_index import democracy_index_avg
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, LinkFeatures, PathFeatures
from bgpsyche.stage3_rank.vectorize_util import one_hot, scale_zero_to_one_linear

PATH_FEATURE_VECTOR_NAMES: t.List[str] = [
    'length',
    'is_valley_free',
    'longest_real_snippet',
    'per_dest_markov_confidence',
]

def vectorize_path_features(features: PathFeatures) -> t.List[t.Union[int, float]]:
    assert -1 <= features['per_dest_markov_confidence'] <= 1

    return [
        scale_zero_to_one_linear(features['length'], val_min=0, val_max=10),

        int(features['is_valley_free'] or False),

        scale_zero_to_one_linear(
            features['longest_real_snippet'], val_min=0, val_max=10
        ),

        features['per_dest_markov_confidence'],
    ]

LINK_FEATURE_VECTOR_NAMES: t.List[str] = [
    'rel_p2c',
    'rel_p2p',
    'rel_c2p',
    'rel_unknown',
    'distance_km',
    'trade_factor',
    'confidence_from_seen_count',
]

_one_hot_rels = one_hot(['p2p', 'c2p', 'p2c'], optional=True)

def vectorize_link_features(features: LinkFeatures) -> t.List[t.Union[int, float]]:
    return [
        *_one_hot_rels(features['rel']),

        scale_zero_to_one_linear(
            features['distance_km'],
            val_min=0, val_max=40_075, # earth circumference
        ),

        features['trade_factor'],
        features['confidence_from_seen_count'],
    ]

AS_FEATURE_VECTOR_NAMES: t.List[str] = [
    'as_rank_cone',
    'rirstat_born',
    'rirstat_addr_count_v4',
    'rirstat_addr_count_v6',
    'distance_from_path_beginning_km',
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
    'country_democracy_index',
]

_one_hot_as_category = one_hot(list(AS_CATEGORY), optional=False)

def vectorize_as_features(features: ASFeaturesRaw) -> t.List[t.Union[int, float]]:
    return [
        scale_zero_to_one_linear(
            math.log(features['as_rank_cone'] or 1),
            val_min=1,
            val_max=math.log(ASRANK_CUSTOMER_CONE_SIZE_RANGE[1]),
        ),

        scale_zero_to_one_linear(
            mktime(features['rirstat_born'].timetuple())
            if features['rirstat_born'] is not None else 0,
            val_min=0,
            val_max=datetime.combine(
                datetime.today(), datetime.min.time()
            ).timestamp()
        ),

        scale_zero_to_one_linear(
            features['rirstat_addr_count_v4']
            if features['rirstat_addr_count_v4'] is not None else 0,
            val_min=0, val_max=32,
        ),

        scale_zero_to_one_linear(
            features['rirstat_addr_count_v6']
            if features['rirstat_addr_count_v6'] is not None else 0,
            val_min=0, val_max=64,
        ),

        scale_zero_to_one_linear(
            features['distance_from_path_beginning_km'],
            val_min=0, val_max=40_075, # earth circumference
        ),

        *_one_hot_as_category(
            'Unknown' if features['as_category'] == 'Route Collector'
            else features['as_category'],
        ),

        scale_zero_to_one_linear(
            features['country_democracy_index'] or democracy_index_avg(),
            val_min=0, val_max=10,
        ),
    ]