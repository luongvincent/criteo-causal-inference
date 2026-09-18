"""Policy curve: incremental conversions captured vs. fraction of users targeted."""

import pandas as pd


def conservative_decile_table(decile_tbl: pd.DataFrame) -> pd.DataFrame:
    """Zero out actual_ate for deciles whose CI includes zero - statistically
    indistinguishable from no effect, so don't credit them with incremental conversions."""
    tbl = decile_tbl.copy()
    not_significant = (tbl["ci_low"] <= 0) & (tbl["ci_high"] >= 0)
    tbl.loc[not_significant, "actual_ate"] = 0.0
    return tbl


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
