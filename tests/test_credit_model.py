import json
import os
from pathlib import Path

import numpy as np
import pytest

import credit_model as cm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "german.data"


def test_data_shape_and_default_rate():
    X, y = cm.load(DATA)
    assert len(X) == 1000 and y.mean() == pytest.approx(0.30)
    for col in cm.EXCLUDED:          # fair-lending exclusions really are dropped
        assert col not in X.columns


def test_ks_statistic():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert cm.ks_stat(y, np.array([.1, .2, .3, .7, .8, .9])) == pytest.approx(1.0)   # perfect split
    assert cm.ks_stat(y, np.array([.5] * 6)) == pytest.approx(0.0)                   # no information


def test_expected_cost_matches_cost_matrix():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    # threshold 0.5: reject idx 1 (good -> cost 1) and idx 3 (bad, correct); approve idx 2 (bad -> cost 5)
    assert cm.expected_cost(y, p, 0.5) == pytest.approx((1 + 5) / 4)
    # approving everyone costs 5 x the default rate
    assert cm.expected_cost(y, np.zeros(4), 0.5) == pytest.approx(5 * 0.5)


def test_pipeline_reproduces_published_results(tmp_path, monkeypatch):
    published = json.loads((ROOT / "results" / "results.json").read_text())
    monkeypatch.chdir(tmp_path)
    cm.main(str(DATA))
    fresh = json.loads((tmp_path / "results" / "results.json").read_text())
    for model in ("logistic_scorecard", "gradient_boosting"):
        assert fresh["models"][model]["test"] == published["models"][model]["test"]
        assert fresh["models"][model]["test_at_cost_threshold"] == published["models"][model]["test_at_cost_threshold"]
    assert fresh["scorecard"]["base_points"] == published["scorecard"]["base_points"]
