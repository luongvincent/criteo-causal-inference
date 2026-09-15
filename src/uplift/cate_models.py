"""Meta-learners for CATE estimation."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from uplift.balance import FEATURES


def _new_gbm(**kwargs) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0, **kwargs)


# --- S-learner: one model, treatment as a feature ---

def fit_s_learner(train: pd.DataFrame, outcome: str, features: list[str] = FEATURES,
                   treatment_col: str = "treatment", **gbm_kwargs) -> HistGradientBoostingClassifier:
    cols = features + [treatment_col]
    model = _new_gbm(**gbm_kwargs)
    model.fit(train[cols], train[outcome])
    return model


def predict_s_learner(model: HistGradientBoostingClassifier, df: pd.DataFrame,
                       features: list[str] = FEATURES, treatment_col: str = "treatment"):
    cols = features + [treatment_col]

    as_treated = df[features].copy()
    as_treated[treatment_col] = 1
    as_control = df[features].copy()
    as_control[treatment_col] = 0

    p1 = model.predict_proba(as_treated[cols])[:, 1]
    p0 = model.predict_proba(as_control[cols])[:, 1]
    return p1 - p0


# --- T-learner: two separate models, one per group ---

def fit_t_learner(train: pd.DataFrame, outcome: str, features: list[str] = FEATURES,
                   treatment_col: str = "treatment", **gbm_kwargs):
    treated = train[train[treatment_col] == 1]
    control = train[train[treatment_col] == 0]

    model_treated = _new_gbm(**gbm_kwargs).fit(treated[features], treated[outcome])
    model_control = _new_gbm(**gbm_kwargs).fit(control[features], control[outcome])
    return model_treated, model_control


def predict_t_learner(models, df: pd.DataFrame, features: list[str] = FEATURES):
    model_treated, model_control = models
    p1 = model_treated.predict_proba(df[features])[:, 1]
    p0 = model_control.predict_proba(df[features])[:, 1]
    return p1 - p0
