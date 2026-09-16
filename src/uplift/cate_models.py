"""Meta-learners for CATE estimation."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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


# --- X-learner: cross-imputed effects, blended by propensity ---

def fit_x_learner(train: pd.DataFrame, outcome: str, features: list[str] = FEATURES,
                   treatment_col: str = "treatment", **gbm_kwargs):
    # stage 1: reuse T-learner's outcome models
    model_treated, model_control = fit_t_learner(train, outcome, features, treatment_col, **gbm_kwargs)

    treated = train[train[treatment_col] == 1]
    control = train[train[treatment_col] == 0]

    # stage 2: cross-impute individual effects using the *other* group's model
    d1 = treated[outcome].to_numpy() - model_control.predict_proba(treated[features])[:, 1]
    d0 = model_treated.predict_proba(control[features])[:, 1] - control[outcome].to_numpy()

    # stage 3: fit new models on the imputed effects (regressors - targets are continuous)
    tau1 = HistGradientBoostingRegressor(random_state=0).fit(treated[features], d1)
    tau0 = HistGradientBoostingRegressor(random_state=0).fit(control[features], d0)

    # propensity is a constant here since treatment was unconditionally randomized (~0.85)
    propensity = train[treatment_col].mean()

    return {"tau1": tau1, "tau0": tau0, "propensity": propensity}


def predict_x_learner(model: dict, df: pd.DataFrame, features: list[str] = FEATURES):
    tau1_pred = model["tau1"].predict(df[features])
    tau0_pred = model["tau0"].predict(df[features])
    e = model["propensity"]
    return e * tau0_pred + (1 - e) * tau1_pred
