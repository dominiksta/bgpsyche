from datetime import datetime
import logging
import multiprocessing
from os import cpu_count
import statistics
import typing as t

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import bgpsyche.logging_config
from bgpsyche.caching.pickle import PickleFileCache
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates
from bgpsyche.stage2_enrich.enrich import enrich_path
from bgpsyche.stage3_rank.vectorize_features import FEATURE_VECTOR_NAMES, vectorize_features
from bgpsyche.util.benchmark import Progress
from bgpsyche.util.const import JOBLIB_MEMORY
from bgpsyche.service.ext import routeviews, ripe_ris

_LOG = logging.getLogger(__name__)

_WORKER_PROCESSES_AMNT = (cpu_count() or 3) - 2
_WORKER_CHUNKSIZE = 10


def _get_path_candidates_worker(path: t.List[int]):
    return get_path_candidates(path[0], path[-1]), path


def _make_path_dataset(
        candidates_per_real_path = 10,
        real_paths_n = 100,
        routeviews_dts: t.List[datetime] = [
            datetime.fromisoformat('2023-05-01T00:00'),
        ],
        ripe_ris_dts: t.List[datetime] = [
            datetime.fromisoformat('2023-05-01T00:00'),
        ],
) -> t.Tuple[np.ndarray, np.ndarray]:
    real_paths: t.List[t.List[int]] = []
    X, y = [], []

    _LOG.info('Loading paths into memory...')

    for dt in routeviews_dts:
        for path_meta in routeviews.iter_paths(dt, eliminate_path_prepending=True):
            real_paths.append(path_meta['path'])

    for dt in ripe_ris_dts:
        for path_meta in ripe_ris.iter_paths(dt, eliminate_path_prepending=True):
            real_paths.append(path_meta['path'])
            

    _LOG.info('Done loading paths into memory')
    _LOG.info(f'Shuffling paths and taking {real_paths_n}')

    np.random.shuffle(real_paths)
    real_paths = real_paths[:real_paths_n]

    prg = Progress(
        int(len(real_paths) / _WORKER_CHUNKSIZE),
        'Mixing real paths with false candidates'
    )

    with multiprocessing.Pool(_WORKER_PROCESSES_AMNT) as p:
        iter = 0
        for w_resp in p.imap_unordered(
                _get_path_candidates_worker, real_paths,
                chunksize=_WORKER_CHUNKSIZE
        ):
            iter += 1
            resp, path = w_resp
            np.random.shuffle(resp['candidates'])
            iter_candidates = 0
            for candidate in resp['candidates']:
                if candidate == path: continue
                iter_candidates += 1
                if iter_candidates >= candidates_per_real_path: break
                X.append(vectorize_features(enrich_path(candidate)))
                y.append(0)

            X.append(vectorize_features(enrich_path(path)))
            y.append(1)

            if iter % _WORKER_CHUNKSIZE == 0:
                prg.update()

        prg.complete()


    return np.array(X), np.array(y)
        

def test():
    rng = np.random.RandomState(42)

    # TODO: try other classifiers and *auto-sklearn*
    classifier = DecisionTreeClassifier(random_state=rng)

    cache = PickleFileCache(
        'candidate_classifier_training_dataset',
        _make_path_dataset,
    )
    # cache.invalidate()
    X, y = cache.get()
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