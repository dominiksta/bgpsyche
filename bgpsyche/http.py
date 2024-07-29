import typing as t

from flask import Flask, Response, request, jsonify
from bgpsyche.stage1_candidates.get_candidates import get_path_candidates
from bgpsyche.stage3_rank.classifier_nn import make_prediction_function
from bgpsyche.stage3_rank.vectorize_features import vectorize_as_features
from bgpsyche.util.benchmark import bench_function

flask_app = Flask('bgpsyche_listen')

_PredictFun = t.Callable[[t.List[t.List[int]]], t.List[float]]

_PREDICT_FUN: _PredictFun

@bench_function
def init():
    global _PREDICT_FUN
    _PREDICT_FUN = make_prediction_function(retrain=False)
    # init caches
    get_path_candidates(3320, 6939)
    _PREDICT_FUN([[3320, 6939]])

@flask_app.route("/predict")
def predict():
    source = request.args.get('source', type=int)
    if source is None: return '<source> arg is missing', 400
    sink = request.args.get('sink', type=int)
    if sink is None: return '<sink> arg is missing', 400

    if source == sink: return Response('source and sink are the same', status=400)

    candidates = get_path_candidates(source, sink)
    if len(candidates) == 0: return Response('no paths found', status=404)

    probs = _PREDICT_FUN(candidates)
    assert len(probs) == len(candidates)
    paths_with_probs = [ (candidates[i], probs[i]) for i in range(len(probs)) ]
    paths_with_probs.sort(key=lambda el: el[1], reverse=True)

    return jsonify(paths_with_probs[:20])