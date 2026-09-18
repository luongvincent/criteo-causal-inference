"""Average treatment effect estimation via difference in means, with CIs."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


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
