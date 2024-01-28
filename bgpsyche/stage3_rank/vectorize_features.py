import typing as t

import numpy as np
from bgpsyche.stage2_enrich.types import ASFeaturesRaw, PathFeatures

PATH_FEATURE_VECTOR_NAMES: t.List[str] = [
    'length',
    'is_valley_free',
    'longest_real_snippet_diff',
    # 'asrank_variance',
    # 'geographic_distance_diff',
]

def vectorize_path_features(features: PathFeatures) -> t.List[t.Union[int, float]]:
    return [
        features['length'],
        0 if features['is_valley_free'] is None
        else int(features['is_valley_free']),
        features['longest_real_snippet_diff'],
        # features['asrank_variance'],
        # features['geographic_distance_diff'],
    ]

AS_FEATURE_VECTOR_NAMES: t.List[str] = [
    # 'asn',
    # 'ripe_name',
    # 'ripe_country',
    'as_rank',
    # 'peeringdb_type',
    'peeringdb_prefix_count_v4',
    'peeringdb_prefix_count_v6',
    # 'peeringdb_geographic_scope',
]

def vectorize_as_features(features: ASFeaturesRaw) -> t.List[t.Union[int, float]]:
    return [
        features['as_rank'],
        features['peeringdb_prefix_count_v4'] or -1,
        features['peeringdb_prefix_count_v6'] or -1,
    ]