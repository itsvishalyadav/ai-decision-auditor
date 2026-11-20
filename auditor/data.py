"""Synthetic datasets used to demonstrate the auditor.

The loan dataset deliberately contains *historical bias*: in the past, female
applicants and young applicants were approved less often than equally
qualified male / older applicants. It also contains a *proxy* feature
(``part_time``) that is correlated with gender, so a model can still learn the
bias even if the gender column is removed. The auditor should catch this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "approved"
PROTECTED = ["gender", "age_group"]

NUMERIC_FEATURES = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "debt_to_income",
    "late_payments",
]
CATEGORICAL_FEATURES = ["gender", "education", "part_time", "age_group"]

# Features a person could realistically change (used for counterfactuals).
ACTIONABLE = {
    "income": {"step": 5000, "direction": +1, "min": 0, "max": 400000},
    "credit_score": {"step": 15, "direction": +1, "min": 300, "max": 850},
    "loan_amount": {"step": 5000, "direction": -1, "min": 1000, "max": 1000000},
    "debt_to_income": {"step": 0.03, "direction": -1, "min": 0.0, "max": 1.0},
    "late_payments": {"step": 1, "direction": -1, "min": 0, "max": 20},
    "employment_years": {"step": 1, "direction": +1, "min": 0, "max": 45},
}


def _age_group(age: np.ndarray) -> np.ndarray:
    return np.where(age < 25, "Under 25", np.where(age < 60, "25-59", "60+"))


def make_loan_data(n: int = 4000, seed: int = 42, bias_strength: float = 1.0) -> pd.DataFrame:
    """Generate a synthetic loan-application dataset with injected bias."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age = np.clip(rng.normal(40, 13, n), 18, 75).round().astype(int)
    education = rng.choice(["High School", "Bachelor", "Master", "PhD"], size=n, p=[0.35, 0.40, 0.20, 0.05])
    edu_bonus = pd.Series(education).map({"High School": 0, "Bachelor": 8000, "Master": 15000, "PhD": 22000}).to_numpy()

    # Proxy feature: part-time work is far more common for women in this data.
    part_time = np.where(rng.random(n) < np.where(gender == "Female", 0.45, 0.10), "Yes", "No")

    income = rng.lognormal(10.8, 0.45, n) + edu_bonus - np.where(part_time == "Yes", 12000, 0)
    income = np.clip(income, 12000, 350000).round(-2)
    employment_years = np.clip(np.minimum(age - 18, rng.gamma(2.2, 3.5, n)), 0, 45).round().astype(int)
    credit_score = np.clip(rng.normal(670, 70, n) + employment_years * 1.5, 300, 850).round().astype(int)
    loan_amount = np.clip(rng.lognormal(10.3, 0.6, n), 2000, 500000).round(-2)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 0.9, 0.01, 0.95).round(3)
    late_payments = rng.poisson(np.clip(2.2 - (credit_score - 600) / 100, 0.1, 5)).astype(int)

    # "True" creditworthiness (merit) - what a fair process would look at.
    merit = (
        0.000025 * (income - 45000)
        + 0.012 * (credit_score - 650)
        - 0.000006 * (loan_amount - 30000)
        - 3.0 * (debt_to_income - 0.25)
        - 0.35 * late_payments
        + 0.04 * employment_years
    )

    # Historical decision makers were biased against women and young applicants,
    # and also penalised part-time workers (which is a proxy for gender).
    group = _age_group(age)
    bias = bias_strength * (
        -0.9 * (gender == "Female")
        - 0.6 * (group == "Under 25")
        - 0.5 * (part_time == "Yes")
    )
    logit = merit + bias + 0.4 + rng.normal(0, 0.6, n)
    approved = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "age_group": group,
            "education": education,
            "part_time": part_time,
            "income": income,
            "employment_years": employment_years,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "late_payments": late_payments,
            TARGET: approved,
        }
    )


def make_hiring_decisions(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """A CSV of decisions made by some *external* AI hiring tool.

    Used for the "audit an external AI" mode: the auditor never sees that
    model, only its decisions, the true outcome and the protected attributes.
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    ethnicity = rng.choice(["Group A", "Group B", "Group C"], size=n, p=[0.6, 0.25, 0.15])
    experience = np.clip(rng.gamma(2.5, 2.5, n), 0, 30).round(1)
    skill = np.clip(rng.normal(65, 15, n), 0, 100).round()
    qualified = ((0.05 * skill + 0.25 * experience + rng.normal(0, 1, n)) > 4.6).astype(int)
    tool_score = (
        0.05 * skill + 0.25 * experience + rng.normal(0, 1.1, n)
        - 0.7 * (gender == "Female") - 0.5 * (ethnicity == "Group C")
    )
    shortlisted = (tool_score > 4.6).astype(int)
    return pd.DataFrame(
        {
            "candidate_id": [f"C{i:05d}" for i in range(n)],
            "gender": gender,
            "ethnicity": ethnicity,
            "years_experience": experience,
            "skill_test_score": skill,
            "actually_qualified": qualified,
            "ai_shortlisted": shortlisted,
        }
    )
