"""Training of the AI model that is going to be audited."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

MODEL_TYPES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=0),
}


@dataclass
class TrainedModel:
    """A trained model plus everything the auditor needs to know about it."""

    name: str
    pipeline: Pipeline
    features: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    threshold: float = 0.5
    group_thresholds: dict | None = None  # optional post-processing per group
    group_col: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        """Short fingerprint of the model (name + params + features)."""
        payload = json.dumps(
            {"name": self.name, "features": self.features, "params": str(self.pipeline.get_params()),
             "thr": self.threshold, "gthr": self.group_thresholds, "notes": self.notes},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        Xf = X[self.features].astype(self.X_train[self.features].dtypes.to_dict())
        return self.pipeline.predict_proba(Xf)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        p = self.predict_proba(X)
        if self.group_thresholds and self.group_col in X.columns:
            thr = X[self.group_col].map(self.group_thresholds).fillna(self.threshold).to_numpy()
        else:
            thr = self.threshold
        return (p >= thr).astype(int)

    def performance(self) -> dict:
        p = self.predict_proba(self.X_test)
        yhat = self.predict(self.X_test)
        return {
            "accuracy": float(accuracy_score(self.y_test, yhat)),
            "f1": float(f1_score(self.y_test, yhat)),
            "roc_auc": float(roc_auc_score(self.y_test, p)),
            "approval_rate": float(yhat.mean()),
        }


def build_pipeline(features: list[str], model_type: str) -> Pipeline:
    num = [f for f in features if f in NUMERIC_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    pre = ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    )
    return Pipeline([("prep", pre), ("clf", MODEL_TYPES[model_type]())])


def default_features(include_protected: bool) -> list[str]:
    feats = NUMERIC_FEATURES + ["education", "part_time"]
    if include_protected:
        feats = feats + ["gender"]
    return feats


def train_model(
    df: pd.DataFrame,
    model_type: str = "Gradient Boosting",
    include_protected: bool = True,
    sample_weight: np.ndarray | None = None,
    split=None,
    seed: int = 0,
    features: list[str] | None = None,
) -> TrainedModel:
    """Train a model. ``split`` lets mitigation retrain on the exact same split."""
    features = features or default_features(include_protected)
    if split is None:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[TARGET]), df[TARGET], test_size=0.3, random_state=seed, stratify=df[TARGET]
        )
    else:
        X_train, X_test, y_train, y_test = split
    pipe = build_pipeline(features, model_type)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipe.fit(X_train[features], y_train, **fit_kwargs)
    return TrainedModel(model_type, pipe, features, X_train, X_test, y_train, y_test)
