"""Bias mitigation techniques, so the audit can also recommend a fix.

* Pre-processing  - ``reweighing`` (Kamiran & Calders, 2012): give training
  samples weights so that group and label become statistically independent.
* In-processing   - remove the protected attribute AND its known proxies.
* Post-processing - ``group_thresholds`` (Hardt et al., 2016 style): choose a
  decision threshold per group so that selection rates become equal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TrainedModel, train_model


def reweighing_weights(groups: pd.Series, y: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"g": groups.to_numpy(), "y": y.to_numpy()})
    n = len(df)
    p_g = df.g.value_counts() / n
    p_y = df.y.value_counts() / n
    p_gy = df.groupby(["g", "y"]).size() / n
    w = df.apply(lambda r: p_g[r.g] * p_y[r.y] / p_gy[(r.g, r.y)], axis=1)
    return w.to_numpy()


def mitigate_reweighing(base: TrainedModel, attribute: str) -> TrainedModel:
    split = (base.X_train, base.X_test, base.y_train, base.y_test)
    w = reweighing_weights(base.X_train[attribute], base.y_train)
    m = train_model(None, base.name, include_protected=False, sample_weight=w, split=split)
    m.notes.append(f"Reweighing on '{attribute}', protected attribute removed")
    return m


def mitigate_drop_proxies(base: TrainedModel, attribute: str, proxies: list[str]) -> TrainedModel:
    split = (base.X_train, base.X_test, base.y_train, base.y_test)
    feats = [f for f in base.features if f not in proxies and f != attribute]
    m = train_model(None, base.name, split=split, features=feats)
    m.notes.append(f"Removed '{attribute}' and proxies {proxies}")
    return m


def mitigate_thresholds(base: TrainedModel, attribute: str) -> TrainedModel:
    """Pick per-group thresholds so every group has the overall selection rate."""
    p = base.predict_proba(base.X_train)
    target_rate = float((p >= base.threshold).mean())
    thr = {}
    for g in base.X_train[attribute].unique():
        pg = p[(base.X_train[attribute] == g).to_numpy()]
        thr[g] = float(np.quantile(pg, 1 - target_rate))
    m = TrainedModel(base.name, base.pipeline, base.features, base.X_train, base.X_test,
                     base.y_train, base.y_test, base.threshold, thr, attribute,
                     notes=[f"Group-specific thresholds on '{attribute}': " +
                            ", ".join(f"{k}={v:.2f}" for k, v in thr.items())])
    return m


def find_proxies(X: pd.DataFrame, attribute: str, features: list[str], min_assoc: float = 0.2) -> pd.DataFrame:
    """Features statistically associated with the protected attribute.

    Uses Cramer's V for categorical features and the correlation ratio (eta)
    for numeric features. High association => possible proxy discrimination.
    """
    g = X[attribute]
    rows = []
    for f in features:
        if f == attribute:
            continue
        col = X[f]
        if not pd.api.types.is_numeric_dtype(col):
            ct = pd.crosstab(g, col).to_numpy()
            n = ct.sum()
            expected = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
            chi2 = ((ct - expected) ** 2 / expected).sum()
            k = min(ct.shape) - 1
            assoc = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0
            method = "Cramer's V"
        else:
            grand = col.mean()
            ss_between = sum(len(c) * (c.mean() - grand) ** 2 for _, c in col.groupby(g))
            ss_total = ((col - grand) ** 2).sum()
            assoc = float(np.sqrt(ss_between / ss_total)) if ss_total else 0.0
            method = "Correlation ratio"
        rows.append({"feature": f, "association": round(assoc, 3), "method": method,
                     "possible_proxy": assoc >= min_assoc})
    return pd.DataFrame(rows).sort_values("association", ascending=False).reset_index(drop=True)
