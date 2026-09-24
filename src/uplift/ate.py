"""Average treatment effect estimation via difference in means, with CIs."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from uplift.balance import FEATURES


@dataclass
class ATEResult:
    outcome: str
    n_treated: int
    n_control: int
    rate_treated: float
    rate_control: float
    ate: float
    se: float
    ci_low: float
    ci_high: float
    relative_lift: float


def compute_ate(df, outcome: str, treatment_col: str = "treatment", alpha: float = 0.05) -> ATEResult:
    """Difference in means of `outcome` between treatment and control, with a (1-alpha) CI."""
    treated = df.loc[df[treatment_col] == 1, outcome]
    control = df.loc[df[treatment_col] == 0, outcome]

    n1, n0 = len(treated), len(control)
    p1, p0 = treated.mean(), control.mean()

    ate = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)

    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = ate - z * se, ate + z * se

    return ATEResult(
        outcome=outcome,
        n_treated=n1,
        n_control=n0,
        rate_treated=p1,
        rate_control=p0,
        ate=ate,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        relative_lift=ate / p0 if p0 > 0 else np.nan,
    )


@dataclass
class CUPEDResult:
    outcome: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    theta: np.ndarray
    raw_ate: float
    raw_se: float
    variance_reduction: float  # 1 - (cuped_se / raw_se)**2, fraction of variance removed


def compute_ate_cuped(df, outcome: str, features: list[str] = FEATURES,
                       treatment_col: str = "treatment", alpha: float = 0.05) -> CUPEDResult:
    """CUPED: reduce the ATE's variance (not its bias) using a pre-treatment covariate.

    theta is fit by OLS regressing the (centered) outcome on the (centered) features,
    pooled across both arms - treatment is not used in fitting theta, so under true
    randomization this adjustment cannot introduce bias, only remove variance in Y that
    the features already explain. The point estimate is the difference in means of the
    *adjusted* outcome `Y - theta . (X - mean(X))`, which has the same expectation as Y.

    Caveat specific to this dataset (see notebook 07, Part 1): the balance check found the
    features are not perfectly independent of `treatment` (AUC ~0.51). CUPED's "can't
    introduce bias" guarantee assumes that independence holds. With 12 features fit jointly,
    this is close to the same adjustment as the linear covariate-adjusted ATE already
    computed there - expect the CUPED point estimate to shift similarly, which means this
    call is doing double duty (variance reduction *and* the same bias adjustment), not a
    clean, separate variance-only check.
    """
    X = df[features].to_numpy(np.float64)
    Xc = X - X.mean(axis=0)
    y = df[outcome].to_numpy(np.float64)

    theta, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    y_adj = y - Xc @ theta

    t = df[treatment_col].to_numpy()
    treated_adj, control_adj = y_adj[t == 1], y_adj[t == 0]
    n1, n0 = len(treated_adj), len(control_adj)

    ate = treated_adj.mean() - control_adj.mean()
    se = np.sqrt(treated_adj.var(ddof=1) / n1 + control_adj.var(ddof=1) / n0)

    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = ate - z * se, ate + z * se

    raw = compute_ate(df, outcome, treatment_col, alpha)

    return CUPEDResult(
        outcome=outcome, ate=ate, se=se, ci_low=ci_low, ci_high=ci_high, theta=theta,
        raw_ate=raw.ate, raw_se=raw.se, variance_reduction=1 - (se / raw.se) ** 2,
    )


@dataclass
class MDEResult:
    outcome: str
    se: float
    mde_abs: float
    mde_relative: float


def compute_mde(result: ATEResult, alpha: float = 0.05, power: float = 0.8) -> MDEResult:
    """Minimum detectable effect: smallest true effect this design could catch (1-alpha, given power)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    mde_abs = (z_alpha + z_power) * result.se

    return MDEResult(
        outcome=result.outcome,
        se=result.se,
        mde_abs=mde_abs,
        mde_relative=mde_abs / result.rate_control,
    )
