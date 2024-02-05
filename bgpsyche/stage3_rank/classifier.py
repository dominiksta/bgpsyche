from functools import cmp_to_key
import logging
from pprint import pformat
import statistics
import typing as t
from functools import cmp_to_key
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

from bgpsyche.logging_config import logging_setup
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.make_dataset import make_path_level_dataset
from bgpsyche.stage3_rank.vectorize_features import PATH_FEATURE_VECTOR_NAMES, vectorize_path_features
from bgpsyche.caching.pickle import PickleFileCache

logging_setup()
_LOG = logging.getLogger(__name__)

def _train() -> t.Callable[[t.List[int]], float]:
    def _train():
        rng = np.random.RandomState(42)

        # TODO: try other classifiers and *auto-sklearn*. auto-sklearn only works in
        # python3.9 though.
        classifier = GradientBoostingRegressor(random_state=rng)

        dataset = make_path_level_dataset()
        X = [ p['path_features'] for p in dataset ]
        y = [ int(p['real']) for p in dataset ]
        _LOG.info('Got training data set')

        print(X[0:10], y[0:10])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=rng
        )

        _LOG.info('Training classifier...')
        classifier.fit(X_train, y_train)

        _LOG.info(f'Score: {classifier.score(X_test, y_test)}')
        # f1 = f1_score(classifier.predict(X_test), y_test)
        # prec = precision_score(classifier.predict(X_test), y_test)
        # _LOG.info(f'Classifier score: {score}, F1: {f1}, P: {prec}')

        return classifier

    cache = PickleFileCache('classifier_sklearn', _train)
    cache.invalidate()
    classifier = cache.get()

    def predict(path: t.List[int]) -> float:
        pred = classifier.predict([
            vectorize_path_features(enrich_path(path))
        ])
        if pred[0] != 0: print(pred)
        return pred[0]

    return predict


def _correlations() -> t.Dict[str, float]:
    dataset = make_path_level_dataset()
    X = [ p['path_features'] for p in dataset ]
    y = [ int(p['real']) for p in dataset ]

    return {
        feature_name: statistics.correlation(
            [
                features[PATH_FEATURE_VECTOR_NAMES.index(feature_name)]
                for features in X
            ],
            list(y),
        )
        for feature_name in PATH_FEATURE_VECTOR_NAMES if (
                feature_name != 'geographic_distance_diff'
        )
    }


def predict_candidates(source: int, sink: int) -> t.List[t.Tuple[float, t.List[int]]]:
    ret: t.List[t.Tuple[float, t.List[int]]] = []
    predict = _train()

    for candidate in get_path_candidates(source, sink)['candidates']:
        pred = predict(candidate)
        ret.append((pred, candidate))

    ret.sort(key=cmp_to_key(lambda a, b: a[0] > b[0]))

    return ret



if __name__ == '__main__':

    _LOG.info(f'correlations: \n{pformat(_correlations())}')

    # 39120 3356 58453 9808
    preds = predict_candidates(39120, 9808)

    probs = [ p[0] for p in preds ]
    plt.hist(probs)
    plt.show()

    _LOG.info(pformat(preds[:10]))
