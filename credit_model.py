"""Credit default model on the German Credit dataset (1,000 loan applicants).

Compares a logistic-regression scorecard with gradient boosting, picks an
approval cut-off by expected cost (missing a bad loan costs 5x rejecting a good
one, the cost matrix published with the dataset), and turns the logistic model
into a points-based scorecard.

Usage: python credit_model.py data/german.data
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
COST_FN = 5.0   # approve an applicant who defaults
COST_FP = 1.0   # reject an applicant who would have repaid
# Excluded on fair-lending grounds: sex/marital status and foreign-worker status.
EXCLUDED = ["personal_status_sex", "foreign_worker"]
OUT = Path("results")


def load(path):
    df = pd.read_csv(path)
    y = (df.pop("credit_risk") == 0).astype(int)  # 1 = bad / default
    df = df.drop(columns=EXCLUDED)
    return df, y


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"roc_auc": round(auc, 3), "gini": round(2 * auc - 1, 3),
            "ks": round(ks_stat(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3)}


def expected_cost(y, p, t):
    pred = (p >= t).astype(int)  # 1 = reject
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y)


def best_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    costs = [expected_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": t, "approval_rate": round(float((pred == 0).mean()), 3),
            "bad_recall": round(recall_score(y, pred), 3),
            "bad_precision": round(precision_score(y, pred, zero_division=0), 3),
            "bad_rate_in_approved": round(float(fn / max(fn + tn, 1)), 3),
            "missed_defaults": int(fn), "good_rejected": int(fp),
            "cost_per_applicant": round(expected_cost(y, p, t), 3)}


def build(X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
                             ("num", StandardScaler(), num)])
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    gbm = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                               ("num", "passthrough", num)])),
                    ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                           max_iter=200, random_state=SEED))])
    return {"logistic_scorecard": logit, "gradient_boosting": gbm}


def scorecard(model, base=600, pdo=20, base_odds=50):
    """Points per feature level: score = offset + factor * ln(good/bad odds)."""
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = pre.get_feature_names_out()
    coefs = -clf.coef_[0]  # model predicts bad; flip so higher = safer
    rows = [{"feature": n, "points_per_unit": round(float(factor * c), 1)} for n, c in zip(names, coefs)]
    intercept_pts = offset + factor * (-clf.intercept_[0])
    rows = sorted(rows, key=lambda r: abs(r["points_per_unit"]), reverse=True)
    return round(float(intercept_pts), 1), rows


def main(path):
    OUT.mkdir(exist_ok=True)
    X, y = load(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report = {"n_applicants": len(y), "default_rate": round(float(y.mean()), 3),
              "n_train": len(y_tr), "n_test": len(y_te), "excluded_features": EXCLUDED,
              "cost_matrix": {"approve_bad": COST_FN, "reject_good": COST_FP}, "models": {}}
    test_probs = {}
    for name, model in build(X).items():
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        t = best_threshold(y_tr, oof)  # chosen on training folds only
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        test_probs[name] = p
        report["models"][name] = {"cv_train": metrics(y_tr, oof), "test": metrics(y_te, p),
                                  "test_at_cost_threshold": at_threshold(y_te, p, t),
                                  "test_at_0.5": at_threshold(y_te, p, 0.5)}
        if name == "logistic_scorecard":
            scored_model = model
    # Baseline: approve everyone
    report["baseline_approve_all"] = {"cost_per_applicant": round(COST_FN * float(y_te.mean()), 3),
                                      "bad_rate_in_approved": round(float(y_te.mean()), 3)}
    base_pts, pts = scorecard(scored_model)
    report["scorecard"] = {"base_points": base_pts, "scaling": "600 pts at 50:1 good:bad odds, 20 points to double the odds",
                           "top_attributes": pts[:12]}

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(title="ROC, held-out test (300 applicants)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(fontsize=8)
    grid = np.arange(0.05, 0.95, 0.01)
    for name, p in test_probs.items():
        ax[1].plot(grid, [expected_cost(y_te, p, t) for t in grid], label=name)
    ax[1].axhline(report["baseline_approve_all"]["cost_per_applicant"], ls="--", color="grey", lw=1, label="approve everyone")
    ax[1].set(title="Expected cost per applicant (bad approved = 5, good rejected = 1)",
              xlabel="Reject if P(default) ≥ threshold", ylabel="Cost per applicant")
    ax[1].title.set_fontsize(9)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_and_cost.png", dpi=150)

    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/german.data")
