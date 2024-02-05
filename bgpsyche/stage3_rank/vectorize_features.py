import typing as t
from time import mktime

import numpy as np
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, PathFeatures

# TODO: evaluate new rnn on AS-triplet level

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

AS_FEATURE_VECTOR_NAMES: t.List[str] = [
    # 'ripe_country',
    'as_rank',
    'rirstat_born',
    'rirstat_addr_count_v4',
    'rirstat_addr_count_v6',
]

def vectorize_as_features(features: ASFeaturesRaw) -> t.List[t.Union[int, float]]:
    return [
        features['as_rank'],
        mktime(features['rirstat_born'].timetuple())
        if features['rirstat_born'] is not None else -1,
        features['rirstat_addr_count_v4'] or -1,
        features['rirstat_addr_count_v6'] or -1,
    ]