"""Group-fairness metrics (same definitions as Fairlearn / IBM AIF360).

All functions work only on *decisions*: true labels, predicted labels and the
protected attribute. That means they can audit any AI system, even a black box
we cannot open, as long as we can see what it decided.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Thresholds commonly used in practice / literature.
THRESHOLDS = {
    "demographic_parity_difference": 0.10,  # |difference in selection rate|
    "disparate_impact_ratio": 0.80,          # "four-fifths rule" (US EEOC)
    "equal_opportunity_difference": 0.10,    # |difference in true positive rate|
    "equalized_odds_difference": 0.10,       # max(|TPR diff|, |FPR diff|)
    "predictive_parity_difference": 0.10,    # |difference in precision|
}

METRIC_INFO = {
    "demographic_parity_difference": (
        "Demographic parity",
        "Do all groups get a positive decision at the same rate?",
    ),
    "disparate_impact_ratio": (
        "Disparate impact (80% rule)",
        "Lowest group's selection rate divided by the highest. Below 0.8 is evidence of adverse impact.",
    ),
    "equal_opportunity_difference": (
        "Equal opportunity",
        "Among people who truly deserved a 'yes', does every group get one equally often?",
    ),
    "equalized_odds_difference": (
        "Equalized odds",
        "Are both error types (missed approvals and wrong approvals) equal across groups?",
    ),
    "predictive_parity_difference": (
        "Predictive parity",
        "When the AI says 'yes', is it equally often right for every group?",
    ),
}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def group_table(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group confusion-matrix based statistics."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("g"):
        tp = int(((d.y == 1) & (d.p == 1)).sum())
        fp = int(((d.y == 0) & (d.p == 1)).sum())
        tn = int(((d.y == 0) & (d.p == 0)).sum())
        fn = int(((d.y == 1) & (d.p == 0)).sum())
        rows.append(
            {
                "group": g,
                "count": len(d),
                "selection_rate": d.p.mean(),
                "base_rate": d.y.mean(),
                "accuracy": (tp + tn) / len(d),
                "tpr": _safe_div(tp, tp + fn),
                "fpr": _safe_div(fp, fp + tn),
                "precision": _safe_div(tp, tp + fp),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def fairness_metrics(y_true, y_pred, groups) -> dict:
    t = group_table(y_true, y_pred, groups)
    sr = t["selection_rate"]
    return {
        "demographic_parity_difference": float(sr.max() - sr.min()),
        "disparate_impact_ratio": _safe_div(sr.min(), sr.max()),
        "equal_opportunity_difference": float(t["tpr"].max() - t["tpr"].min()),
        "equalized_odds_difference": float(
            max(t["tpr"].max() - t["tpr"].min(), t["fpr"].max() - t["fpr"].min())
        ),
        "predictive_parity_difference": float(t["precision"].max() - t["precision"].min()),
        "most_favoured_group": str(sr.idxmax()),
        "least_favoured_group": str(sr.idxmin()),
    }


def metric_passes(name: str, value: float) -> bool:
    if np.isnan(value):
        return True
    if name == "disparate_impact_ratio":
        return value >= THRESHOLDS[name]
    return value <= THRESHOLDS[name]


def fairness_findings(y_true, y_pred, groups, attribute: str) -> list[dict]:
    """Turn metrics into human-readable audit findings."""
    m = fairness_metrics(y_true, y_pred, groups)
    findings = []
    for key, (title, question) in METRIC_INFO.items():
        value = m[key]
        ok = metric_passes(key, value)
        findings.append(
            {
                "attribute": attribute,
                "metric": title,
                "key": key,
                "value": round(value, 3),
                "threshold": THRESHOLDS[key],
                "status": "PASS" if ok else "FAIL",
                "question": question,
                "detail": (
                    f"'{m['least_favoured_group']}' is the least favoured group and "
                    f"'{m['most_favoured_group']}' the most favoured."
                ),
            }
        )
    return findings


def fairness_score(findings: list[dict]) -> float:
    """0-100 score: share of passed checks, softened by how far failures miss."""
    if not findings:
        return 100.0
    scores = []
    for f in findings:
        if f["status"] == "PASS":
            scores.append(1.0)
        elif f["key"] == "disparate_impact_ratio":
            scores.append(max(0.0, f["value"] / f["threshold"]) * 0.6)
        else:
            scores.append(max(0.0, 1 - f["value"] / (3 * f["threshold"])) * 0.6)
    return round(100 * float(np.mean(scores)), 1)
