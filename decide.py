"""Score one loan application and explain the decision.

A lender can't just say "the model says no": when an application is declined, the applicant
must be told the main reasons (for example the adverse-action notices required by US fair-lending
rules, or a bank's own customer-communication policy). This tool returns what an underwriter or a
loan-origination system needs for one applicant:

  * decision (approve / decline) at the cost-based cut-off learned in credit_model.py
  * probability of default and a points score (600 = 50:1 good:bad odds, +20 points doubles the odds)
  * the top reasons in plain language: the characteristics that push this applicant's risk up the
    most compared with an average applicant

Usage:
    python decide.py examples/applicant_approve.json
    python decide.py examples/applicant_decline.json --cutoff 0.25     # a looser policy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import credit_model as cm

ROOT = Path(__file__).resolve().parent
FACTOR = 20 / np.log(2)
OFFSET = 600 - FACTOR * np.log(50)

# Plain-language reason for each characteristic (used when it raises risk).
REASONS = {
    "status": "Balance or history of the existing checking account",
    "duration": "Long repayment period requested",
    "credit_history": "Past repayment record",
    "purpose": "Purpose of the loan",
    "amount": "Size of the loan requested",
    "savings": "Low or no savings",
    "employment_duration": "Short time in current employment",
    "installment_rate": "Instalment is a high share of disposable income",
    "other_debtors": "No co-applicant or guarantor",
    "present_residence": "Time at current address",
    "property": "No property or other collateral",
    "other_installment_plans": "Other instalment plans already running",
    "housing": "Housing situation (renting or living rent-free)",
    "number_credits": "Number of existing credits at this bank",
    "job": "Job category",
    "people_liable": "Number of dependants",
    "telephone": "No registered telephone",
}
NEVER_A_REASON = {"age"}   # legally sensitive: kept in the model (see README) but never cited against an applicant


def fit():
    """Refit the published logistic scorecard exactly as credit_model.py does (same split and seed)."""
    X, y = cm.load(ROOT / "data" / "german.data")
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=cm.SEED)
    model = cm.build(X)["logistic_scorecard"].fit(X_tr, y_tr)
    return model, X_tr


def _source_column(encoded_name, columns):
    """'cat__savings_... < 100 DM' -> 'savings'; 'num__age' -> 'age'."""
    name = encoded_name.split("__", 1)[1]
    return max((c for c in columns if name == c or name.startswith(c + "_")), key=len)


def contributions(model, frame):
    """Per-applicant, per-characteristic contribution to the log-odds of default."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Z = pre.transform(frame)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    names = pre.get_feature_names_out()
    cols = [_source_column(n, frame.columns) for n in names]
    per_feature = {}
    for j, c in enumerate(cols):
        per_feature[c] = per_feature.get(c, 0) + Z[:, j] * clf.coef_[0][j]
    return per_feature


def validate(applicant, X_ref):
    errors = []
    for col in X_ref.columns:
        if col not in applicant:
            errors.append(f"missing field '{col}'")
        elif X_ref[col].dtype.kind in "iuf":
            if not isinstance(applicant[col], (int, float)):
                errors.append(f"'{col}' must be a number")
        elif applicant[col] not in set(X_ref[col]):
            errors.append(f"'{col}' must be one of: {sorted(set(X_ref[col]))}")
    return errors


def decide(applicant, model=None, X_tr=None, cutoff=None, n_reasons=4):
    if model is None:
        model, X_tr = fit()
    errors = validate(applicant, X_tr)
    if errors:
        raise ValueError("; ".join(errors))
    if cutoff is None:
        published = json.loads((ROOT / "results" / "results.json").read_text())
        cutoff = published["models"]["logistic_scorecard"]["test_at_cost_threshold"]["threshold"]

    row = X_tr.iloc[:0].copy()
    row.loc[0] = [applicant[c] for c in X_tr.columns]
    row = row.astype(X_tr.dtypes.to_dict())
    pd_ = float(model.predict_proba(row)[0, 1])
    score = OFFSET + FACTOR * np.log((1 - pd_) / pd_)

    # reasons: characteristics where this applicant is riskier than the average training applicant
    mine = {c: float(v[0]) for c, v in contributions(model, row).items()}
    avg = {c: float(np.mean(v)) for c, v in contributions(model, X_tr).items()}
    worse = sorted(((mine[c] - avg[c], c) for c in mine if c not in NEVER_A_REASON and mine[c] > avg[c]),
                   reverse=True)
    reasons = [{"characteristic": c, "reason": REASONS[c], "value": applicant[c],
                "points_lost_vs_average": round(float(FACTOR * gap), 1)} for gap, c in worse[:n_reasons]]
    return {"decision": "decline" if pd_ >= cutoff else "approve",
            "probability_of_default": round(pd_, 3),
            "score": round(float(score)),
            "cutoff_probability": cutoff,
            "main_reasons" if pd_ >= cutoff else "main_risk_factors": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("applicant_json")
    ap.add_argument("--cutoff", type=float, help="decline if P(default) >= cutoff (default: the cost-based cut-off)")
    args = ap.parse_args(argv)
    applicant = json.loads(Path(args.applicant_json).read_text())
    try:
        print(json.dumps(decide(applicant, cutoff=args.cutoff), indent=2))
    except ValueError as e:
        sys.exit(f"Invalid application: {e}")


if __name__ == "__main__":
    main()
