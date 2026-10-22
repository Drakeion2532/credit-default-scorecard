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


# ---------------------------------------------------------------- applicant decisions
import decide as d


@pytest.fixture(scope="module")
def fitted():
    return d.fit()


def _load(name):
    return json.loads((ROOT / "examples" / f"applicant_{name}.json").read_text())


def test_examples_get_the_expected_decisions(fitted):
    model, X_tr = fitted
    ok = d.decide(_load("approve"), model, X_tr)
    no = d.decide(_load("decline"), model, X_tr)
    assert ok["decision"] == "approve" and no["decision"] == "decline"
    assert no["probability_of_default"] > ok["probability_of_default"]
    assert no["score"] < ok["score"]                       # lower score = riskier
    assert 1 <= len(no["main_reasons"]) <= 4


def test_score_scale_matches_scorecard_definition(fitted):
    model, X_tr = fitted
    r = d.decide(_load("approve"), model, X_tr)
    p = r["probability_of_default"]
    # 600 points at 50:1 good:bad odds, 20 points per doubling
    assert r["score"] == pytest.approx(600 + 20 * np.log2(((1 - p) / p) / 50), abs=1.5)


def test_age_is_never_given_as_a_reason(fitted):
    model, X_tr = fitted
    for age in (19, 25, 70, 75):
        app = {**_load("decline"), "age": age}
        reasons = d.decide(app, model, X_tr)["main_reasons"]
        assert all(r["characteristic"] != "age" for r in reasons)


def test_cutoff_changes_the_decision(fitted):
    model, X_tr = fitted
    app = _load("decline")
    pd_ = d.decide(app, model, X_tr)["probability_of_default"]
    assert d.decide(app, model, X_tr, cutoff=min(pd_ + 0.01, 0.99))["decision"] == "approve"


def test_invalid_application_is_rejected_with_a_clear_message(fitted):
    model, X_tr = fitted
    with pytest.raises(ValueError, match="savings"):
        d.decide({**_load("approve"), "savings": "lots"}, model, X_tr)
    bad = _load("approve"); del bad["duration"]
    with pytest.raises(ValueError, match="missing field 'duration'"):
        d.decide(bad, model, X_tr)
