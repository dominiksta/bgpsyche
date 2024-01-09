import typing as t

import numpy as np
from bgpsyche.stage2_enrich.types import PathFeatures

FEATURE_VECTOR_NAMES: t.List[str] = [
    'length',
    'is_valley_free',
    'longest_real_snippet_diff',
    'asrank_variance',
    'geographic_distance_diff',
]

def vectorize_features(features: PathFeatures) -> t.List[t.Union[int, float]]:
    return [
        features['length'],
        0 if features['is_valley_free'] is None
        else int(features['is_valley_free']),
        features['longest_real_snippet_diff'],
        features['asrank_variance'],
        features['geographic_distance_diff'],
    ]