"""Policy curve: incremental conversions captured vs. fraction of users targeted."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift.evaluation import decile_table


def conservative_decile_table(decile_tbl: pd.DataFrame) -> pd.DataFrame:
    """Zero out actual_ate for deciles whose CI includes zero - statistically
    indistinguishable from no effect, so don't credit them with incremental conversions."""
    tbl = decile_tbl.copy()
    not_significant = (tbl["ci_low"] <= 0) & (tbl["ci_high"] >= 0)
    tbl.loc[not_significant, "actual_ate"] = 0.0
    return tbl


def select_then_remeasure(df: pd.DataFrame, predicted_cate, outcome: str, n_bins: int = 10,
                          alpha: float = 0.05, seed: int = 0):
    """Split-sample fix for the winner's curse in conservative_decile_table.

    Splits `df` 50/50 (stratified by treatment). Half A decides which deciles are
    significant (CI excludes zero at level `alpha`/`n_bins`, Bonferroni). Half B, whose
    noise is independent of that choice, supplies the actual_ate credited to the
    selected deciles; unselected deciles are zeroed. Deciles are cut separately in each
    half from predicted CATE alone (no outcome information), so boundaries match closely.

    Returns (selection_tbl, measure_tbl, remeasured_tbl). Feed remeasured_tbl to
    policy_curve; note n is half of df, so profits are on a half-sized population.
    """
    idx = np.arange(len(df))
    idx_a, idx_b = train_test_split(idx, test_size=0.5, stratify=df["treatment"], random_state=seed)
    cate = np.asarray(predicted_cate)

    selection_tbl = decile_table(df.iloc[idx_a], cate[idx_a], outcome,
                                 n_bins=n_bins, alpha=alpha / n_bins)
    measure_tbl = decile_table(df.iloc[idx_b], cate[idx_b], outcome,
                               n_bins=n_bins, alpha=alpha / n_bins)

    not_selected = (selection_tbl["ci_low"] <= 0) & (selection_tbl["ci_high"] >= 0)
    remeasured_tbl = measure_tbl.copy()
    remeasured_tbl.loc[not_selected, "actual_ate"] = 0.0
    return selection_tbl, measure_tbl, remeasured_tbl


def policy_curve(decile_tbl: pd.DataFrame, cost_per_impression: float = 0.01,
                  value_per_conversion: float = 50.0) -> pd.DataFrame:
    """Walk deciles from highest predicted CATE to lowest, computing cumulative
    incremental conversions, cost, benefit, and net profit at each targeting fraction.
    """
    sorted_tbl = decile_tbl.sort_values("decile", ascending=False).reset_index(drop=True)

    incremental_conversions = sorted_tbl["actual_ate"] * sorted_tbl["n"]
    cum_n = sorted_tbl["n"].cumsum()
    cum_incremental_conversions = incremental_conversions.cumsum()
    total_n = sorted_tbl["n"].sum()

    fraction_targeted = cum_n / total_n
    cost = cost_per_impression * cum_n
    benefit = value_per_conversion * cum_incremental_conversions
    net_profit = benefit - cost

    return pd.DataFrame({
        "decile": sorted_tbl["decile"],
        "fraction_targeted": fraction_targeted,
        "cum_n": cum_n,
        "cum_incremental_conversions": cum_incremental_conversions,
        "cost": cost,
        "benefit": benefit,
        "net_profit": net_profit,
    })
