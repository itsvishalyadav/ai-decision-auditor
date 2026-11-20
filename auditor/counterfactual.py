"""Counterfactual analysis.

1. ``find_counterfactual`` - "What is the smallest realistic change that would
   have flipped this decision?" (Wachter et al., 2017). Gives the person
   *recourse*: actionable advice instead of just a 'no'.
2. ``protected_flip_test`` - counterfactual fairness check (Kusner et al.,
   2017): change ONLY the protected attribute (e.g. Male -> Female) and see
   whether the decision changes. If it does, the attribute directly drives
   the decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ACTIONABLE


def find_counterfactual(model, x: pd.Series, max_steps: int = 40) -> dict:
    x0 = x.copy()
    target = 1 - int(model.predict(x0.to_frame().T)[0])
    z = x0.copy()
    actionable = {k: v for k, v in ACTIONABLE.items() if k in model.features}
    path = []
    for _ in range(max_steps):
        p_now = model.predict_proba(z.to_frame().T)[0]
        if int(model.predict(z.to_frame().T)[0]) == target:
            break
        best, best_gain = None, 0.0
        candidates = []
        for f, spec in actionable.items():
            direction = spec["direction"] if target == 1 else -spec["direction"]
            # Try small and larger moves: tree models are flat in places, so a
            # single small step may show no gain even though a bigger one would.
            for mult in (1, 3, 6):
                new = np.clip(z[f] + direction * spec["step"] * mult, spec["min"], spec["max"])
                if new == z[f]:
                    continue
                zc = z.copy()
                zc[f] = round(float(new), 3) if isinstance(z[f], (np.floating, float)) else int(round(new))
                candidates.append((f, zc, mult))
        if not candidates:
            break
        probs = model.predict_proba(pd.DataFrame([c[1] for c in candidates]))
        for (f, zc, mult), p in zip(candidates, probs):
            gain = ((p - p_now) if target == 1 else (p_now - p)) / mult  # gain per unit of effort
            if gain > best_gain:
                best, best_gain = (f, zc), gain
        if best is None:
            break
        z = best[1]
        path.append(best[0])
    success = int(model.predict(z.to_frame().T)[0]) == target
    changes = []
    for f in actionable:
        if z[f] != x0[f]:
            changes.append({"feature": f, "from": x0[f], "to": z[f]})
    return {
        "success": success,
        "target": target,
        "changes": changes,
        "new_probability": float(model.predict_proba(z.to_frame().T)[0]),
        "steps": len(path),
    }


def flip_value(series: pd.Series) -> dict:
    vals = list(series.dropna().unique())
    if len(vals) == 2:
        return {vals[0]: vals[1], vals[1]: vals[0]}
    return {}


def protected_flip_test(model, X: pd.DataFrame, attribute: str) -> dict:
    """Flip the protected attribute for every row and count changed decisions."""
    mapping = flip_value(X[attribute])
    if not mapping:
        return {"attribute": attribute, "tested": False, "reason": "attribute is not binary"}
    Xf = X.copy()
    Xf[attribute] = Xf[attribute].map(mapping)
    p0, p1 = model.predict_proba(X), model.predict_proba(Xf)
    d0, d1 = model.predict(X), model.predict(Xf)
    changed = d0 != d1
    return {
        "attribute": attribute,
        "tested": True,
        "in_model": attribute in model.features,
        "flip_rate": float(changed.mean()),
        "n_changed": int(changed.sum()),
        "mean_abs_prob_shift": float(np.abs(p1 - p0).mean()),
        "max_abs_prob_shift": float(np.abs(p1 - p0).max()),
    }
