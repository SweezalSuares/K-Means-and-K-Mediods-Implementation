"""Smoke tests. Run from the project root:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clustering as cl  # noqa: E402


def _data():
    return pd.read_csv(ROOT / "data" / "CC_GENERAL.csv")


def test_preprocess_shapes_and_no_missing():
    p = cl.preprocess(_data())
    assert p.X_scaled.shape == (8950, 17)
    assert not np.isnan(p.X_scaled).any()
    assert "CUST_ID" not in p.features
    assert np.allclose(p.X_scaled.mean(axis=0), 0, atol=1e-6)


def test_kmeans_k3_matches_notebook():
    p = cl.preprocess(_data())
    labels, centres, _ = cl.fit_kmeans(p.X_scaled, 3, 42)
    assert sorted(np.bincount(labels)) == [1235, 1596, 6119]
    assert round(cl.evaluate(p.X_scaled, labels)["Silhouette"], 3) == 0.251


def test_kmedoids_returns_valid_partition():
    p = cl.preprocess(_data())
    D = cl.distance_matrix(p.X_scaled)
    labels, med, cost = cl.fit_kmedoids(D, 3, 42)
    assert len(set(labels)) == 3 and len(med) == 3 and cost > 0


def test_elbow_k_on_simple_curve():
    assert cl.elbow_k([2, 3, 4, 5, 6], [100, 40, 30, 28, 27]) == 3
