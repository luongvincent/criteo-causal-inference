# criteo-casual-inference

Causal inference practice project on the [Criteo Uplift Prediction dataset](https://huggingface.co/datasets/criteo/criteo-uplift) (Diemert et al., AdKDD 2018) — a randomized ad-targeting experiment with ~14M users. Goal, method, and full results are written up in [`reports/analysis.md`](reports/analysis.md); the project prompt and working constraints are in [`claude.md`](claude.md).

## Do not commit or push

This repo is local only. Do not `git commit` or `git push` any changes to GitHub.

## Notebooks

Developed on a 1M-row sample (`notebooks/01`-`06`), then rerun on the full ~14M-row dataset (`notebooks/07`) for final numbers.

| Notebook | Covers |
|---|---|
| [`01_covariate_balance.ipynb`](notebooks/01_covariate_balance.ipynb) | Standardized mean differences (SMD) between treatment and control on all 12 features — checks that randomization held before trusting anything downstream. |
| [`02_ate_and_power.ipynb`](notebooks/02_ate_and_power.ipynb) | Average treatment effect (ATE) for `visit` and `conversion`, with confidence intervals, plus retrospective minimum detectable effect (MDE) at 80% power. |
| [`03_cate_estimation.ipynb`](notebooks/03_cate_estimation.ipynb) | Conditional average treatment effect (CATE) estimation with a strict train/eval split: S-, T-, X-, and DR-learner meta-learners plus a causal forest, on a gradient-boosted base learner. |
| [`04_evaluation_deciles_qini.ipynb`](notebooks/04_evaluation_deciles_qini.ipynb) | Ranking-based evaluation of the CATE models — Qini curves and AUUC — plus a sleeping-dogs check (deciles where the ad measurably hurts users). |
| [`05_policy_curve.ipynb`](notebooks/05_policy_curve.ipynb) | Policy curve: incremental conversions vs. fraction of users targeted, under assumed economics; a naive version and a statistically-honest (Bonferroni-corrected, split-sample) version, since selecting and measuring "significant" deciles on the same data inflates the estimate. |
| [`06_naive_exposure_comparison.ipynb`](notebooks/06_naive_exposure_comparison.ipynb) | The "do it wrong" comparison: rerunning the ATE using `exposure` (not randomized) instead of `treatment` (randomized), to quantify the bias from conditioning on a post-treatment variable. |
| [`07_full_data_analysis.ipynb`](notebooks/07_full_data_analysis.ipynb) | Full-data (~14M row) rerun: sample ratio mismatch (SRM) check, covariate balance at scale, ATE/MDE, three independent covariate-adjustment checks (linear, cross-fitted doubly-robust, CUPED) after the balance check found a small but real assignment/feature dependence, CATE models, decile/Qini evaluation, and the policy curve under two different cost models. |

## Code

Shared logic lives in `src/uplift/` (balance checks, ATE/CUPED, meta-learners, ranking evaluation, policy curve), imported by every notebook via `sys.path.insert(0, "../src")`. Data loaders cache the 1M-row sample and the full dataset as parquet under `data/` (gitignored) after the first download from Hugging Face.
