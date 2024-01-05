import logging
import statistics
import typing as t

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import bgpsyche.logging_config
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.make_dataset import make_path_dataset
from bgpsyche.stage3_rank.vectorize_features import FEATURE_VECTOR_NAMES, vectorize_features

_LOG = logging.getLogger(__name__)

def test():
    rng = np.random.RandomState(42)

    # TODO: try other classifiers and *auto-sklearn*
    classifier = DecisionTreeClassifier(random_state=rng)

    X, y = make_path_dataset()
    _LOG.info('Got training data set')

    print(X[0:10], y[0:10])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=rng
    )

    _LOG.info('Training classifier...')
    classifier.fit(X_train, y_train)

    _LOG.info('Scoring classifier...')
    score = classifier.score(X_test, y_test)
    _LOG.info(f'Classifier score: {score}')


    print(abs(statistics.correlation(
        [
            features[FEATURE_VECTOR_NAMES.index('geographic_distance_diff')]
            for features in X
        ],
        list(y)
    )))


    def classify(path: t.List[int]) -> bool:
        return classifier.predict([
            vectorize_features(enrich_path(path))
        ]) == 1

    # first path of each block is correct, others have random manually edited
    # mutations
    test_paths = [
        [2497, 3356, 749],
        [2497, 11164, 749],
        [2497, 3320, 749],

        [2497, 11164, 2152, 2152, 7377],
        [2497, 11164, 3320, 2152, 7377],
        [2497, 11164, 2152, 7377],

        [14840, 6939, 4637, 45177],
        [14840, 6939, 1234, 45177],
        [14840, 6939, 45177],
    ]

    for path in test_paths:
        _LOG.info(f'Predict {path}: {classify(path)}')


if __name__ == '__main__': test()