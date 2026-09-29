import time

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.data import load_and_split_data


@pytest.fixture(scope="module")
def gate_model():
    X_train, X_test, y_train, y_test = load_and_split_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_metric_threshold_gate(gate_model):
    model, X_test, y_test = gate_model

    predictions = model.predict(X_test)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    assert macro_f1 >= 0.88


def test_inference_latency_gate(gate_model):
    model, X_test, _ = gate_model

    # Warm-up prediction
    model.predict(X_test)

    start_time = time.perf_counter()
    model.predict(X_test)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert latency_ms <= 30


def test_output_schema_integrity(gate_model):
    model, X_test, _ = gate_model

    predictions = model.predict(X_test)

    assert set(np.unique(predictions)).issubset({0, 1, 2})
