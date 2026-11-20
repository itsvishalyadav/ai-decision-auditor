"""Explainability: why did the AI make this decision?

* ``global_importance`` - permutation feature importance (Breiman 2001): how
  much does model accuracy drop if we shuffle one feature?
* ``shapley_explanation`` - a model-agnostic Monte-Carlo approximation of
  Shapley values (Strumbelj & Kononenko 2014; the idea behind SHAP, Lundberg &
  Lee 2017). It splits one prediction into a contribution per feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def global_importance(model, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X))
    rows = []
    for f in model.features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        rows.append({"feature": f, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def shapley_explanation(
    model, x: pd.Series, background: pd.DataFrame, n_permutations: int = 150, seed: int = 0
) -> pd.DataFrame:
    """Approximate Shapley values for one row ``x``.

    For random feature orderings we start from a random background person and
    switch features to the applicant's values one at a time; the change in
    approval probability caused by each switch is that feature's contribution.
    Contributions add up (approximately) to: prediction - average prediction.
    """
    rng = np.random.default_rng(seed)
    feats = model.features
    d = len(feats)
    bg = background[feats].reset_index(drop=True)
    rows = []
    orders = []
    for _ in range(n_permutations):
        order = rng.permutation(d)
        z = bg.iloc[len(orders) % len(bg)].copy()  # cycle background rows evenly
        rows.append(z.copy())
        for j in order:
            z[feats[j]] = x[feats[j]]
            rows.append(z.copy())
        orders.append(order)
    batch = pd.DataFrame(rows)
    for c in feats:  # keep dtypes consistent with training data
        batch[c] = batch[c].astype(background[c].dtype)
    preds = model.predict_proba(batch).reshape(n_permutations, d + 1)
    contrib = np.zeros(d)
    for k, order in enumerate(orders):
        diffs = np.diff(preds[k])
        contrib[order] += diffs
    contrib /= n_permutations
    out = pd.DataFrame(
        {"feature": feats, "value": [x[f] for f in feats], "contribution": contrib}
    )
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def explanation_text(expl: pd.DataFrame, decision: int, top: int = 3) -> str:
    """Plain-language reason sentence (like an 'adverse action notice')."""
    if decision == 1:
        rel = expl[expl.contribution > 0].head(top)
        verb = "Main reasons for APPROVAL"
    else:
        rel = expl[expl.contribution < 0].head(top)
        verb = "Main reasons for REJECTION"
    if rel.empty:
        return f"{verb}: no single dominant factor."
    parts = [f"{r.feature} = {r.value} ({r.contribution:+.1%})" for r in rel.itertuples()]
    return f"{verb}: " + "; ".join(parts)
