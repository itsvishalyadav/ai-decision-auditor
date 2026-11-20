"""Robustness / stability checks.

A trustworthy model should not change its decision when the input changes by
a tiny, meaningless amount (e.g. income 52,000 vs 52,300). We add small random
noise to the numeric features and measure how many decisions flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NUMERIC_FEATURES

NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10]  # as a fraction of each feature's std-dev
MAX_FLIP_RATE_AT_2PCT = 0.03  # pass criterion


def noise_stability(model, X: pd.DataFrame, levels=NOISE_LEVELS, n_trials: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num = [f for f in NUMERIC_FEATURES if f in model.features]
    base = model.predict(X)
    stds = X[num].std()
    rows = []
    for lvl in levels:
        rates = []
        for _ in range(n_trials):
            Xn = X.copy()
            for f in num:
                Xn[f] = Xn[f] + rng.normal(0, lvl * stds[f], len(Xn))
            rates.append(float((model.predict(Xn) != base).mean()))
        rows.append({"noise_level": lvl, "flip_rate": float(np.mean(rates))})
    return pd.DataFrame(rows)


def robustness_score(table: pd.DataFrame) -> float:
    r = float(table.loc[table.noise_level == 0.02, "flip_rate"].iloc[0])
    # 0% flips -> 100, at the pass limit (3%) -> ~67, at 3x the limit -> 0
    return round(100 * max(0.0, 1 - r / (3 * MAX_FLIP_RATE_AT_2PCT)), 1)


def boundary_cases(model, X: pd.DataFrame, margin: float = 0.05) -> pd.DataFrame:
    """Decisions that sit very close to the threshold - candidates for human review."""
    p = model.predict_proba(X)
    mask = np.abs(p - model.threshold) < margin
    out = X[mask].copy()
    out["probability"] = p[mask]
    return out
