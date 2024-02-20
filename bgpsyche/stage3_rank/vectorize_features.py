import typing as t
from time import mktime

import numpy as np
from bgpsyche.stage2_enrich.democracy_index import democracy_index_avg
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, LinkFeatures, PathFeatures

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
    'distance_km',
    'trade_service_volume_million_usd',
]

def vectorize_link_features(features: LinkFeatures) -> t.List[t.Union[int, float]]:
    return [
        int(features['rel'] == 'p2c'),
        int(features['rel'] == 'p2p'),
        int(features['rel'] == 'c2p'),
        int(features['rel'] is None),
        features['distance_km'],
        features['trade_service_volume_million_usd'],
    ]

AS_FEATURE_VECTOR_NAMES: t.List[str] = [
    'country_democracy_index',
    'as_rank',
    'rirstat_born',
    'rirstat_addr_count_v4',
    'rirstat_addr_count_v6',
]

def vectorize_as_features(features: ASFeaturesRaw) -> t.List[t.Union[int, float]]:
    return [
        features['country_democracy_index'] or democracy_index_avg(),
        features['as_rank'],
        mktime(features['rirstat_born'].timetuple())
        if features['rirstat_born'] is not None else -1,
        features['rirstat_addr_count_v4'] or -1,
        features['rirstat_addr_count_v6'] or -1,
    ]