from pathlib import Path

import numpy as np

from arboles_decision.pipeline import (
    FEATURES,
    evaluate_model,
    fit_model,
    gini_impurity,
    load_data,
    split_data,
    weighted_gini,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "riesgo_clinico_simulado.csv"


def test_gini_known_cases():
    assert gini_impurity([0, 0, 0]) == 0.0
    assert np.isclose(gini_impurity([0, 0, 1, 1]), 0.5)
    assert np.isclose(weighted_gini([0, 0], [1, 1]), 0.0)


def test_split_is_reproducible_and_disjoint():
    frame = load_data(DATA)
    first = split_data(frame)
    second = split_data(frame)
    x_train, x_test, y_train, y_test = first
    assert x_train.index.equals(second[0].index)
    assert x_test.index.equals(second[1].index)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert len(x_train) + len(x_test) == len(frame)
    assert y_train.index.equals(x_train.index)
    assert y_test.index.equals(x_test.index)


def test_model_shape_and_metrics_are_valid():
    frame = load_data(DATA)
    x_train, x_test, y_train, y_test = split_data(frame)
    model = fit_model(x_train, y_train)
    metrics, prediction, probability = evaluate_model(model, x_test, y_test)
    assert model.get_depth() <= 3
    assert model.n_features_in_ == len(FEATURES)
    assert len(prediction) == len(y_test)
    assert np.all((probability >= 0) & (probability <= 1))
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
