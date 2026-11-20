"""Runs every check and combines them into a Trust Scorecard."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .counterfactual import protected_flip_test
from .explain import global_importance, shapley_explanation
from .fairness import fairness_findings, fairness_score, group_table
from .mitigation import find_proxies
from .robustness import MAX_FLIP_RATE_AT_2PCT, boundary_cases, noise_stability, robustness_score

PILLARS = {
    # pillar: (weight, EU AI Act reference, NIST AI RMF characteristic)
    "Fairness": (0.35, "Art. 10 - data & bias examination", "Fair - with harmful bias managed"),
    "Transparency": (0.20, "Art. 13 - transparency to users", "Explainable & interpretable"),
    "Robustness": (0.15, "Art. 15 - accuracy, robustness", "Safe, secure & resilient"),
    "Performance": (0.15, "Art. 15 - accuracy", "Valid & reliable"),
    "Accountability": (0.15, "Art. 12 & 14 - logging, human oversight", "Accountable & transparent"),
}


def _verdict(score: float, critical: bool) -> str:
    if score >= 80 and not critical:
        return "TRUSTWORTHY"
    if score >= 60:
        return "CONDITIONAL - fix issues before deployment"
    return "NOT TRUSTWORTHY - do not deploy"


def explanation_fidelity(model, X: pd.DataFrame, n: int = 8, seed: int = 0) -> float:
    """Check the explanations are faithful: contributions must add up to
    (prediction - average prediction). Returns mean absolute error."""
    rng = np.random.default_rng(seed)
    bg = X.sample(min(100, len(X)), random_state=seed)
    mean_pred = model.predict_proba(bg).mean()
    errs = []
    for i in rng.choice(len(X), size=min(n, len(X)), replace=False):
        x = X.iloc[i]
        e = shapley_explanation(model, x, bg, n_permutations=60, seed=int(i))
        errs.append(abs(e.contribution.sum() - (model.predict_proba(x)[0] - mean_pred)))
    return float(np.mean(errs))


def run_full_audit(model, protected: list[str], ledger=None, fast: bool = False) -> dict:
    X, y = model.X_test, model.y_test
    yhat = model.predict(X)
    result = {"model_name": model.name, "model_id": model.model_id, "notes": list(model.notes),
              "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_test": len(X)}

    # 1. Performance
    perf = model.performance()
    result["performance"] = perf
    perf_score = round(min(100.0, max(0.0, (perf["roc_auc"] - 0.5) / 0.4 * 100)), 1)

    # 2. Fairness (per protected attribute)
    findings, tables, flips, proxies, fscores = [], {}, {}, {}, []
    for attr in protected:
        f = fairness_findings(y, yhat, X[attr], attr)
        findings += f
        fscores.append(fairness_score(f))
        tables[attr] = group_table(y, yhat, X[attr])
        flips[attr] = protected_flip_test(model, X, attr)
        proxies[attr] = find_proxies(X, attr, model.features)
    fair_score = float(np.mean(fscores)) if fscores else 100.0
    for attr, ft in flips.items():  # direct discrimination penalty
        if ft.get("tested") and ft["flip_rate"] > 0.01:
            fair_score -= min(20, 200 * ft["flip_rate"])
    fair_score = round(max(0.0, fair_score), 1)
    result.update(fairness_findings=findings, group_tables=tables, flip_tests=flips, proxies=proxies)

    # 3. Transparency
    imp = global_importance(model, X, y, n_repeats=2 if fast else 5)
    fidelity_err = explanation_fidelity(model, X, n=4 if fast else 8)
    top3 = list(imp.feature.head(3))
    protected_in_top = [a for a in protected if a in top3]
    proxy_in_top = [f for attr in protected for f in proxies[attr].query("possible_proxy").feature if f in top3]
    trans_score = 100 - min(40, fidelity_err * 400) - 25 * len(protected_in_top) - 10 * len(proxy_in_top)
    trans_score = round(max(0.0, trans_score), 1)
    result.update(importance=imp, fidelity_error=fidelity_err,
                  protected_in_top=protected_in_top, proxy_in_top=proxy_in_top)

    # 4. Robustness
    rob = noise_stability(model, X, n_trials=2 if fast else 5)
    rob_score = robustness_score(rob)
    border = boundary_cases(model, X)
    result.update(robustness=rob, boundary_cases=border)

    # 5. Accountability
    chain = ledger.verify() if ledger is not None else {"valid": True, "reason": "no ledger attached"}
    n_logged = len(ledger.records()) if ledger is not None else 0
    acc_score = 100.0 if chain["valid"] else 0.0
    result.update(ledger_status=chain, n_logged=n_logged)

    scores = {"Fairness": fair_score, "Transparency": trans_score, "Robustness": rob_score,
              "Performance": perf_score, "Accountability": acc_score}
    overall = round(sum(scores[p] * PILLARS[p][0] for p in PILLARS), 1)
    critical = any(f["key"] == "disparate_impact_ratio" and f["status"] == "FAIL" for f in findings) \
        or not chain["valid"] or any(v.get("tested") and v["flip_rate"] > 0.05 for v in flips.values())
    result.update(scores=scores, overall=overall, critical=critical, verdict=_verdict(overall, critical))
    result["recommendations"] = recommendations(result, protected)
    return result


def recommendations(r: dict, protected: list[str]) -> list[str]:
    recs = []
    failed = [f for f in r["fairness_findings"] if f["status"] == "FAIL"]
    for attr in protected:
        fa = [f for f in failed if f["attribute"] == attr]
        if fa:
            recs.append(f"Bias against '{fa[0]['detail'].split(chr(39))[1]}' on '{attr}': "
                        f"{len(fa)} of 5 fairness checks fail. Apply reweighing or group thresholds "
                        f"(see Mitigation tab) and re-audit.")
        ft = r["flip_tests"].get(attr, {})
        if ft.get("tested") and ft.get("in_model") and ft["flip_rate"] > 0.01:
            recs.append(f"'{attr}' is used directly by the model: changing only {attr} flips "
                        f"{ft['flip_rate']:.1%} of decisions. Remove it from the inputs.")
        px = r["proxies"][attr].query("possible_proxy")
        if len(px):
            recs.append(f"Possible proxy features for '{attr}': {', '.join(px.feature)}. "
                        f"Removing '{attr}' alone will not remove the bias.")
    if r["robustness"].loc[r["robustness"].noise_level == 0.02, "flip_rate"].iloc[0] > MAX_FLIP_RATE_AT_2PCT:
        recs.append("Model is unstable under small input noise; use a simpler / regularised model.")
    if len(r["boundary_cases"]):
        recs.append(f"{len(r['boundary_cases'])} test decisions are within 5% of the threshold; "
                    f"route such borderline cases to a human reviewer (human oversight).")
    if not r["ledger_status"]["valid"]:
        recs.append(f"AUDIT LOG TAMPERED at record {r['ledger_status']['broken_at']}: investigate immediately.")
    if r["performance"]["roc_auc"] < 0.75:
        recs.append("Predictive performance is weak; improve the model before deployment.")
    if not recs:
        recs.append("No major issues found. Keep monitoring decisions in production.")
    return recs
