"""Covariate balance checks between treatment and control groups."""

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]


def smd(df: pd.DataFrame, features: list[str] = FEATURES, treatment_col: str = "treatment") -> pd.Series:
    """Standardized mean difference (treated - control) for each feature, pooled std in the denominator."""
    treated = df.loc[df[treatment_col] == 1, features]
    control = df.loc[df[treatment_col] == 0, features]

    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)

    return mean_diff / pooled_std
