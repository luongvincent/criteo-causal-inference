"""Evaluate CATE models by ranking, not per-user error."""

import pandas as pd

from uplift.ate import compute_ate


def decile_table(df: pd.DataFrame, predicted_cate, outcome: str,
                  treatment_col: str = "treatment", n_bins: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Bin held-out users by predicted CATE, report the actual measured ATE within each bin.

    Decile 0 = lowest predicted CATE, decile n_bins-1 = highest. `alpha` sets each
    bin's CI level - pass alpha/n_bins (Bonferroni correction) to control the
    overall false-positive rate across all n_bins simultaneous tests, not just one.
    """
    work = df.copy()
    work["predicted_cate"] = predicted_cate
    # rank first, then cut, so tied/duplicate predicted values don't break qcut into fewer bins
    work["decile"] = pd.qcut(work["predicted_cate"].rank(method="first"), n_bins, labels=False)

    rows = []
    for d in range(n_bins):
        bucket = work[work["decile"] == d]
        result = compute_ate(bucket, outcome, treatment_col, alpha=alpha)
        rows.append({
            "decile": d,
            "n": len(bucket),
            "mean_predicted_cate": bucket["predicted_cate"].mean(),
            "actual_ate": result.ate,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        })

    return pd.DataFrame(rows)
