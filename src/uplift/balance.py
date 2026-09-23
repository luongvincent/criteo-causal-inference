"""Covariate balance checks between treatment and control groups."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = [f"f{i}" for i in range(12)]


@dataclass
class SRMResult:
    n_treated: int
    n_control: int
    expected_treated: float
    expected_control: float
    chi2: float
    p_value: float
    flagged: bool  # p_value < alpha


def sample_ratio_mismatch(df: pd.DataFrame, expected_treated_share: float = 0.85,
                           treatment_col: str = "treatment", alpha: float = 1e-3) -> SRMResult:
    """Chi-square goodness-of-fit test: does the observed treatment/control split match the
    intended assignment ratio? This is a different check from `smd` - it tests whether the
    *assignment mechanism* worked (e.g. a logging bug dropping some control events), not
    whether the two groups look similar on covariates. `alpha` defaults to 1e-3 (stricter
    than the usual 0.05), the common convention for SRM checks since they run continuously
    and a true SRM calls the whole experiment's validity into question.
    """
    n_treated = int((df[treatment_col] == 1).sum())
    n_control = int((df[treatment_col] == 0).sum())
    n_total = n_treated + n_control

    expected_treated = n_total * expected_treated_share
    expected_control = n_total * (1 - expected_treated_share)

    chi2, p_value = stats.chisquare(
        f_obs=[n_treated, n_control],
        f_exp=[expected_treated, expected_control],
    )

    return SRMResult(
        n_treated=n_treated, n_control=n_control,
        expected_treated=expected_treated, expected_control=expected_control,
        chi2=chi2, p_value=p_value, flagged=p_value < alpha,
    )


def smd(df: pd.DataFrame, features: list[str] = FEATURES, treatment_col: str = "treatment") -> pd.Series:
    """Standardized mean difference (treated - control) for each feature, pooled std in the denominator."""
    treated = df.loc[df[treatment_col] == 1, features]
    control = df.loc[df[treatment_col] == 0, features]

    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)

    return mean_diff / pooled_std
