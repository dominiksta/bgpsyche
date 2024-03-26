from functools import cmp_to_key, lru_cache
import logging
from pprint import pformat
from random import randint
import statistics
import typing as t
from functools import cmp_to_key

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from bgpsyche.logging_config import logging_setup
from bgpsyche.stage1_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.make_dataset import make_dataset
from bgpsyche.stage3_rank.vectorize_features import (
    PATH_FEATURE_VECTOR_NAMES, vectorize_path_features
)
from bgpsyche.caching.pickle import PickleFileCache

logging_setup()
_LOG = logging.getLogger(__name__)

@lru_cache()
def _train() -> t.Callable[[t.List[int]], float]:
    def _train():
        rng = np.random.RandomState(42)

        # we could try other classifiers and *auto-sklearn*. auto-sklearn only
        # works in python3.9 though.
        classifier = RandomForestClassifier(random_state=rng)

        dataset = make_dataset()
        X = [ p['path_features'] for p in dataset ]
        y = [ int(p['real']) for p in dataset ]
        _LOG.info('Got training data set')

        length = [ p['path_features'][0] for p in dataset ]
        print(statistics.correlation(length, y))

        is_vf = [ p['path_features'][1] for p in dataset ]
        print(statistics.correlation(is_vf, y))
        is_vf_real = [ is_vf[i] for i in range(len(y)) if y[i] == 1 ]
        is_vf_not_real = [ is_vf[i] for i in range(len(y)) if y[i] == 1 ]
        print(is_vf_real.count(1) / len(is_vf_real))
        print(is_vf_not_real.count(0) / len(is_vf_not_real))

        # X = []
        # y = []

        # bool_chance = lambda p: int(randint(0, 100) <= p)

        # for _ in range(10_000):
        #     is_vf = bool_chance(90)
        #     X.append([is_vf])
        #     y.append(1)

        #     for __ in range(3):
        #         is_vf = bool_chance(50)
        #         X.append([is_vf])
        #         y.append(0)

        # print(statistics.correlation([ x[0] for x in X ], y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=rng
        )

        _LOG.info('Training classifier...')
        classifier.fit(X_train, y_train)

        _LOG.info(f'Score: {classifier.score(X_test, y_test)}')
        pred_test = classifier.predict(X_test)
        _LOG.info(
            f'F1={f1_score(y_test, pred_test)} ' +
            f'ACC={accuracy_score(y_test, pred_test)} ' +
            f'PREC={precision_score(y_test, pred_test)} ' +
            f'REC={recall_score(y_test, pred_test)}'
        )
        pred_train = classifier.predict(X_train)
        _LOG.info(f'Score: {classifier.score(X_train, y_train)}')
        _LOG.info(
            f'F1={f1_score(y_train, pred_train)} ' +
            f'ACC={accuracy_score(y_train, pred_train)} ' +
            f'PREC={precision_score(y_train, pred_train)} ' +
            f'REC={recall_score(y_train, pred_train)}'
        )
        # f1 = f1_score(classifier.predict(X_test), y_test)
        # prec = precision_score(classifier.predict(X_test), y_test)
        # _LOG.info(f'Classifier score: {score}, F1: {f1}, P: {prec}')

        return classifier

    cache = PickleFileCache('classifier_sklearn', _train)
    cache.invalidate()
    classifier = cache.get()

    def predict(path: t.List[int]) -> float:
        pred = classifier.predict_proba([
            vectorize_path_features(enrich_path(path))
        ])
        # if pred[0] != 0: print(pred)
        return pred[0]

    return predict


# def _correlations() -> t.Dict[str, float]:
#     dataset = make_dataset()
#     X = [ p['path_features'] for p in dataset ]
#     y = [ int(p['real']) for p in dataset ]

#     return {
#         feature_name: statistics.correlation(
#             [
#                 features[PATH_FEATURE_VECTOR_NAMES.index(feature_name)]
#                 for features in X
#             ],
#             list(y),
#         )
#         for feature_name in PATH_FEATURE_VECTOR_NAMES if (
#                 feature_name != 'geographic_distance_diff'
#         )
#     }


def predict_candidates(source: int, sink: int) -> t.List[t.Tuple[float, t.List[int]]]:
    ret: t.List[t.Tuple[float, t.List[int]]] = []
    predict = _train()

    for candidate in get_path_candidates(source, sink):
        pred = predict(candidate)
        ret.append((pred, candidate))

    ret.sort(key=cmp_to_key(lambda a, b: a[0] > b[0]))

    return ret


def predict_probs(
        paths: t.List[t.List[int]],
        retrain = True,
) -> t.List[float]:
    predict = _train()
    ret = [ predict(path) for path in paths ]
    ret.sort(reverse=True)
    return ret



if __name__ == '__main__':

    # _LOG.info(f'correlations: \n{pformat(_correlations())}')

    _train()

    # 39120 3356 58453 9808
    # candidates = get_path_candidates(39120, 9808)
    # preds = predict_probs(candidates)

    # # probs = [ p[0] for p in preds ]
    # # plt.hist(probs)
    # # plt.show()

    # _LOG.info(pformat([
    #     (preds[i], candidates[i]) for i in range(10)
    # ]))
