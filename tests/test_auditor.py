import warnings

import numpy as np
import pandas as pd
import pytest

from auditor.audit import run_full_audit
from auditor.counterfactual import find_counterfactual, protected_flip_test
from auditor.data import make_loan_data
from auditor.explain import shapley_explanation
from auditor.fairness import fairness_metrics
from auditor.ledger import AuditLedger
from auditor.mitigation import mitigate_reweighing, reweighing_weights
from auditor.model import train_model

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return train_model(make_loan_data(n=2000), "Logistic Regression")


def test_fairness_metrics_known_values():
    y = [1, 1, 0, 0, 1, 1, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0]
    g = ["A"] * 4 + ["B"] * 4
    m = fairness_metrics(y, p, g)
    assert m["demographic_parity_difference"] == pytest.approx(0.75 - 0.25)
    assert m["disparate_impact_ratio"] == pytest.approx(0.25 / 0.75)
    assert m["equal_opportunity_difference"] == pytest.approx(1.0 - 0.5)


def test_perfectly_fair_decisions_pass():
    y = [1, 0] * 50
    m = fairness_metrics(y, y, ["A", "B"] * 25 + ["B", "A"] * 25)
    assert m["demographic_parity_difference"] == 0
    assert m["disparate_impact_ratio"] == 1


def test_auditor_detects_injected_bias(model):
    r = run_full_audit(model, ["gender"], fast=True)
    di = [f for f in r["fairness_findings"] if f["key"] == "disparate_impact_ratio"][0]
    assert di["status"] == "FAIL"
    assert r["fairness_findings"][0]["detail"].startswith("'Female'")
    assert "part_time" in set(r["proxies"]["gender"].query("possible_proxy").feature)


def test_no_bias_when_history_is_fair():
    m = train_model(make_loan_data(n=3000, bias_strength=0.0), "Logistic Regression", include_protected=False)
    r = run_full_audit(m, ["gender"], fast=True)
    di = [f for f in r["fairness_findings"] if f["key"] == "disparate_impact_ratio"][0]
    assert di["value"] > 0.8


def test_shapley_values_add_up(model):
    x = model.X_test.iloc[0]
    bg = model.X_train.sample(100, random_state=0)
    e = shapley_explanation(model, x, bg, n_permutations=200)  # 2 passes over 100 background rows
    assert e.contribution.sum() == pytest.approx(model.predict_proba(x)[0] - model.predict_proba(bg).mean(), abs=0.03)


def test_counterfactual_flips_decision(model):
    rejected = model.X_test[model.predict(model.X_test) == 0].iloc[0]
    cf = find_counterfactual(model, rejected)
    assert cf["success"]
    assert all(c["feature"] not in ("gender", "age") for c in cf["changes"])


def test_flip_test(model):
    assert protected_flip_test(model, model.X_test, "gender")["flip_rate"] > 0
    blind = train_model(make_loan_data(n=2000), "Logistic Regression", include_protected=False)
    assert protected_flip_test(blind, blind.X_test, "gender")["flip_rate"] == 0


def test_reweighing_makes_label_independent_of_group():
    g = pd.Series(["A"] * 80 + ["B"] * 20)
    y = pd.Series([1] * 60 + [0] * 20 + [1] * 5 + [0] * 15)
    w = reweighing_weights(g, y)
    df = pd.DataFrame({"g": g, "y": y, "w": w})
    rates = df.groupby("g").apply(lambda d: np.average(d.y, weights=d.w), include_groups=False)
    assert rates["A"] == pytest.approx(rates["B"])


def test_mitigation_improves_fairness(model):
    before = run_full_audit(model, ["gender"], fast=True)
    after = run_full_audit(mitigate_reweighing(model, "gender"), ["gender"], fast=True)
    di = lambda r: [f["value"] for f in r["fairness_findings"] if f["key"] == "disparate_impact_ratio"][0]
    assert di(after) > di(before)


def test_ledger_detects_tampering(tmp_path, model):
    L = AuditLedger(tmp_path / "log.jsonl")
    for i in range(4):
        L.append(model.model_id, {"i": i}, i % 2, 0.5, "test")
    assert L.verify()["valid"]
    L.tamper(2)
    v = L.verify()
    assert not v["valid"] and v["broken_at"] == 2
