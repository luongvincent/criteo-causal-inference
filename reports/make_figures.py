"""Regenerate the report figures in reports/figures/ from the full ~14M-row data.

Run from anywhere:  python reports/make_figures.py
Needs the cached full dataset (data/raw/criteo_full.parquet) and the cached CATE
predictions written by notebooks/07_full_data_analysis.ipynb (data/raw/full_cates.pkl).
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.ate import compute_ate, compute_ate_cuped  # noqa: E402
from uplift.balance import FEATURES, smd  # noqa: E402
from uplift.data import FULL_PATH, load_full_cached  # noqa: E402
from uplift.evaluation import decile_table, qini_curve  # noqa: E402
from uplift.policy import policy_curve  # noqa: E402

OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# colour-blind-safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, PINK, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7f7f7f", "#D55E00"
MODEL_COLORS = {"S-learner": GREY, "T-learner": ORANGE, "X-learner": GREEN, "DR-learner": BLUE}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name, flush=True)


df = load_full_cached()
cates = pickle.load(open(FULL_PATH.parent / "full_cates.pkl", "rb"))
eval_ = df.loc[cates["eval_index"]]

# ------------------------------------------------------------------ 1. covariate balance
n1, n0 = (df["treatment"] == 1).sum(), (df["treatment"] == 0).sum()
se_smd = np.sqrt(1 / n1 + 1 / n0)
b = smd(df).sort_values(key=abs)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.axvspan(-3 * se_smd, 3 * se_smd, color=GREEN, alpha=0.25,
           label=f"noise expected under true randomization (±3 SE = ±{3 * se_smd:.4f})")
ax.barh(b.index, b.values, color=BLUE, height=0.65)
for x in (-0.1, 0.1):
    ax.axvline(x, color=RED, ls="--", lw=1.2)
ax.axvline(0, color="black", lw=0.8)
ax.plot([], [], color=RED, ls="--", label="conventional |SMD| < 0.1 threshold")
ax.set_xlim(-0.12, 0.12)
ax.set_xlabel("standardized mean difference (treated − control)")
ax.set_title("Every feature passes the 0.1 rule, yet sits far outside sampling noise\n"
             f"(largest SMD ≈ {b.abs().max():.3f}, about {b.abs().max() / se_smd:.0f} standard errors from zero)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
save(fig, "01_covariate_balance.png")

# ------------------------------------------------------------------ 2. raw vs adjusted ATE
X = df[FEATURES].to_numpy(np.float64).copy()
X = X - X.mean(axis=0)
t = df["treatment"].to_numpy(np.float64)
Z = np.column_stack([np.ones(len(df)), t, X, X * t[:, None]])
ZtZ_inv = np.linalg.inv(Z.T @ Z)

# cross-fitted DR values are copied from notebooks/07_full_data_analysis.ipynb (seeded run,
# 4M-row subsample); they are slow to recompute here.
DR = {"visit": (0.007342, 0.000238), "conversion": (0.001075, 0.000072)}

est = {}
for outcome in ["visit", "conversion"]:
    y = df[outcome].to_numpy(np.float64)
    beta = ZtZ_inv @ (Z.T @ y)
    resid = y - Z @ beta
    meat = (Z * resid[:, None]).T @ (Z * resid[:, None])
    se_lin = np.sqrt((ZtZ_inv @ meat @ ZtZ_inv)[1, 1])
    raw = compute_ate(df, outcome)
    cup = compute_ate_cuped(df, outcome)
    est[outcome] = [("raw difference in means", raw.ate, raw.se),
                    ("linear adjustment", beta[1], se_lin),
                    ("doubly robust (4M-row subsample)", *DR[outcome]),
                    ("CUPED", cup.ate, cup.se)]
del Z, X

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, outcome in zip(axes, ["visit", "conversion"]):
    rows = est[outcome]
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f" if outcome == "visit" else "%.5f"))
    labels = [r[0] for r in rows][::-1]
    for i, (_, m, se) in enumerate(rows[::-1]):
        color = RED if i == len(rows) - 1 else BLUE
        ax.errorbar(m, i, xerr=1.96 * se, fmt="o", color=color, capsize=4, ms=6)
        ax.annotate(f"{m:.5f}", (m, i), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5)
    ax.axvline(rows[0][1], color=RED, ls=":", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("estimated ATE (95% CI)")
    ax.set_title(f"{outcome}: adjusted estimates sit {'well ' if outcome == 'visit' else ''}below the raw one")
fig.suptitle("Adjusting for the features lowers the estimated effect; for visit, all three methods agree", y=1.03)
save(fig, "02_adjusted_ate.png")

# ------------------------------------------------------------------ 3. decile bars
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for r, outcome in enumerate(["visit", "conversion"]):
    overall = compute_ate(eval_, outcome).ate
    for c, model in enumerate(["DR-learner", "X-learner"]):
        ax = axes[r, c]
        d = decile_table(eval_, cates[outcome][model], outcome, alpha=0.05 / 10)
        err = [d["actual_ate"] - d["ci_low"], d["ci_high"] - d["actual_ate"]]
        colors = [BLUE if k < 9 else ORANGE for k in d["decile"]]
        ax.bar(d["decile"], d["actual_ate"], yerr=err, color=colors, capsize=3, error_kw={"lw": 1})
        ax.axhline(overall, color=RED, ls="--", lw=1, label=f"overall ATE ({overall:.4f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{outcome} — {model}")
        ax.set_xticks(range(10))
        ax.legend(loc="upper left", fontsize=8.5)
        if c == 0:
            ax.set_ylabel("measured ATE in decile")
        if r == 1:
            ax.set_xlabel("decile of predicted CATE (0 = lowest, 9 = highest)")
fig.suptitle("Full data (5.6M held-out users): the effect is concentrated in the top predicted decile\n"
             "bars = measured treated-vs-control difference; whiskers = 99.5% CI (Bonferroni, 10 deciles)", y=1.0)
fig.tight_layout()
save(fig, "03_decile_ate.png")

# ------------------------------------------------------------------ 4. Qini curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, outcome in zip(axes, ["visit", "conversion"]):
    for model, color in MODEL_COLORS.items():
        q = qini_curve(eval_, cates[outcome][model], outcome)
        ax.plot(q["fraction"], q["qini"], color=color, label=model, lw=1.8)
    ax.plot(q["fraction"], q["random"], color="black", ls="--", lw=1, label="random targeting")
    ax.set_title(outcome)
    ax.set_xlabel("fraction of users targeted (highest predicted CATE first)")
    ax.set_ylabel("cumulative incremental outcomes (eval set)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="lower right", fontsize=8.5)
fig.suptitle("Qini curves: targeting the top of the ranking beats random on visit; on conversion the S-learner cannot rank", y=1.02)
save(fig, "04_qini_curves.png")

# ------------------------------------------------------------------ 5. policy curve, two cost models
e_overall = eval_.loc[eval_["treatment"] == 1, "exposure"].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, model in zip(axes, ["DR-learner", "X-learner"]):
    tbl = decile_table(eval_, cates["conversion"][model], "conversion")
    for label, cost, color in [("cost billed on every targeted user ($0.01)", 0.01, BLUE),
                               (f"cost billed only on delivered impressions ($0.01 × {e_overall:.3f})", 0.01 * e_overall, ORANGE)]:
        curve = policy_curve(tbl, cost_per_impression=cost)
        ax.plot(curve["fraction_targeted"], curve["net_profit"] / 1000, marker="o", color=color, label=label)
    ax.set_title(f"conversion — {model}")
    ax.set_xlabel("fraction of users targeted (highest predicted CATE first)")
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.set_ylim(0, 320)
axes[0].set_ylabel("net profit, thousands of USD (eval set)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8.5)
fig.suptitle("Policy curve: the top decile carries most of the value; going further depends on how the ads are billed\n"
             "(illustrative economics: 50 USD per conversion, 0.01 USD per impression; not Criteo's real numbers)", y=1.04)
save(fig, "05_policy_curve.png")

# ------------------------------------------------------------------ 6. exposure vs treatment
fig, ax = plt.subplots(figsize=(7, 4.4))
w = 0.36
for i, outcome in enumerate(["visit", "conversion"]):
    correct = compute_ate(df, outcome, "treatment")
    naive = compute_ate(df, outcome, "exposure")
    ax.bar(i - w / 2, correct.ate, w, color=BLUE, label="correct: split by randomized treatment" if i == 0 else None)
    ax.bar(i + w / 2, naive.ate, w, color=RED, label="naive: split by ad exposure" if i == 0 else None)
    ax.annotate(f"{correct.ate:.4f}", (i - w / 2, correct.ate), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.annotate(f"{naive.ate:.3f}\n({naive.ate / correct.ate:.0f}x)", (i + w / 2, naive.ate),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
ax.set_yscale("log")
ax.set_ylim(5e-4, 1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(["visit", "conversion"])
ax.set_ylabel("estimated effect (log scale)")
ax.set_title("Conditioning on exposure instead of treatment overstates the effect ~37-46x")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "06_exposure_vs_treatment.png")
