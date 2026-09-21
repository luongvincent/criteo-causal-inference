"""Meta-learners for CATE estimation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from uplift.balance import FEATURES

_E_CLIP = 0.01  # keep e(x) away from 0/1 so inverse-propensity weights stay bounded


def _new_gbm(**kwargs) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=0, **kwargs)


# --- Propensity: P(treatment=1 | features) ---
# Treatment was randomized, but notebook 07 finds a small, real dependence on the features
# (AUC ~0.51 predicting assignment), so a constant propensity is not exactly right.

def _fit_propensity(train: pd.DataFrame, features: list[str], treatment_col: str) -> HistGradientBoostingClassifier:
    return _new_gbm().fit(train[features], train[treatment_col])


def _crossfit_propensity(train: pd.DataFrame, features: list[str], treatment_col: str) -> np.ndarray:
    """Out-of-fold e(x): 2 folds, each row scored by the model fit on the other half."""
    fold = np.arange(len(train)) % 2
    e = np.zeros(len(train))
    for k in (0, 1):
        model = _new_gbm().fit(train.loc[fold != k, features], train.loc[fold != k, treatment_col])
        e[fold == k] = model.predict_proba(train.loc[fold == k, features])[:, 1]
    return np.clip(e, _E_CLIP, 1 - _E_CLIP)


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

    # propensity is estimated from the features, not assumed constant: notebook 07 shows
    # assignment is not exactly independent of them (see _fit_propensity)
    propensity = _fit_propensity(train, features, treatment_col)

    return {"tau1": tau1, "tau0": tau0, "propensity": propensity}


def predict_x_learner(model: dict, df: pd.DataFrame, features: list[str] = FEATURES):
    tau1_pred = model["tau1"].predict(df[features])
    tau0_pred = model["tau0"].predict(df[features])
    e = np.clip(model["propensity"].predict_proba(df[features])[:, 1], _E_CLIP, 1 - _E_CLIP)
    return e * tau0_pred + (1 - e) * tau1_pred


# --- DR-learner: doubly robust pseudo-outcome, one final smoothing model ---

def fit_dr_learner(train: pd.DataFrame, outcome: str, features: list[str] = FEATURES,
                    treatment_col: str = "treatment", **gbm_kwargs) -> HistGradientBoostingRegressor:
    t = train[treatment_col].to_numpy()
    y = train[outcome].to_numpy()

    # cross-fit the outcome models: each row's mu1/mu0 come from T-learner models fit on the
    # other half, so residuals (y - mu) aren't shrunk by the models having seen that row
    fold = np.arange(len(train)) % 2
    mu1 = np.zeros(len(train))
    mu0 = np.zeros(len(train))
    for k in (0, 1):
        model_treated, model_control = fit_t_learner(train[fold != k], outcome, features, treatment_col, **gbm_kwargs)
        mu1[fold == k] = model_treated.predict_proba(train.loc[fold == k, features])[:, 1]
        mu0[fold == k] = model_control.predict_proba(train.loc[fold == k, features])[:, 1]

    # estimated propensity e(x), cross-fitted so each row's e comes from a model that never saw it
    e = _crossfit_propensity(train, features, treatment_col)

    # doubly robust pseudo-outcome: outcome-model estimate + propensity-weighted residual correction
    phi = (mu1 - mu0) + (t / e) * (y - mu1) - ((1 - t) / (1 - e)) * (y - mu0)

    return HistGradientBoostingRegressor(random_state=0).fit(train[features], phi)


def predict_dr_learner(model: HistGradientBoostingRegressor, df: pd.DataFrame, features: list[str] = FEATURES):
    return model.predict(df[features])


# --- Causal forest: honest, tree-based, splits directly on effect heterogeneity ---

def fit_causal_forest(train: pd.DataFrame, outcome: str, features: list[str] = FEATURES,
                       treatment_col: str = "treatment", n_estimators: int = 200):
    from econml.dml import CausalForestDML

    model = CausalForestDML(
        model_y=HistGradientBoostingRegressor(random_state=0),
        model_t=HistGradientBoostingClassifier(random_state=0),
        discrete_treatment=True,
        honest=True,
        n_estimators=n_estimators,
        random_state=0,
    )
    model.fit(Y=train[outcome], T=train[treatment_col], X=train[features])
    return model


def predict_causal_forest(model, df: pd.DataFrame, features: list[str] = FEATURES):
    return model.effect(df[features])
