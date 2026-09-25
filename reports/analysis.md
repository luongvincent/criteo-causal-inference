# Causal inference on the Criteo Uplift dataset

Source: Criteo Uplift Prediction dataset (Diemert et al., AdKDD 2018), ~14M rows, one row per user. `treatment` (eligibility to be targeted) is randomized at ~85/15. `exposure` (whether an ad was actually shown) is not — it's the outcome of Criteo's real-time auction, which selects users it predicts are valuable. Outcomes: `visit` (~4.7% base rate) and `conversion` (~0.29% base rate). Twelve anonymized, randomly-projected features (`f0`-`f11`).

Numbers below are from the full ~14M-row dataset (`notebooks/07_full_data_analysis.ipynb`), not the 1M-row development sample, except where a section says otherwise (the `exposure` comparison in Section 4 and the causal forest in Section 6 are from the 1M-row sample).

## 1. Question and identification

The question: does ad targeting cause visits and conversions, and if so, for whom, and who should be targeted given a budget?

The identification argument rests entirely on randomization. Criteo withheld a random ~15% of users from targeting. Because that split was random, treatment and control differ, in expectation, only in whether they were eligible to be targeted — not in any prior characteristic. That makes a simple difference in outcome means a valid estimate of the causal effect (the average treatment effect, ATE), with no adjustment needed *if randomization held exactly*. Sections 3 and 5 check how well it did.

This is an intent-to-treat analysis: it uses `treatment` (eligibility), not `exposure` (whether an ad was actually shown). `exposure` is decided *after* treatment assignment, by an auction that selects for users predicted to convert. Conditioning on it — comparing exposed users to everyone else — throws away the randomization and re-introduces exactly the selection problem the experiment exists to solve. Section 4 quantifies how badly.

`visit` and `conversion` are treated as two independent outcomes, each with its own treatment effect, not as a funnel — conditioning on `visit` to explain `conversion` would repeat the same post-treatment-conditioning error at a different step.

## 2. Methodology

### 2.1 Pipeline

1. **Assignment checks.** A sample ratio mismatch (SRM) test (does the observed treatment/control split match the intended 85/15?), then standardized mean differences (SMD) on all 12 features and a classifier test (can the features predict `treatment`?), to check how well randomization held before trusting anything downstream. These catch different failure modes.
2. **ATE.** Difference in means for `visit` and `conversion`, with 95% CIs, plus retrospective minimum detectable effect (MDE) at 80% power. Because step 1 found a small feature imbalance, this is cross-checked with three covariate-adjusted estimates: linear regression adjustment, a cross-fitted doubly-robust estimate, and CUPED.
3. **CATE estimation**, with a strict train/eval split (never evaluate a model on data used to fit it): S-learner (foil), T-learner, X-learner, DR-learner, all on a gradient-boosted base learner. A causal forest was fit on the 1M-row development sample but does not scale to the full ~14M rows, so it's excluded from the full-data numbers below.
4. **Evaluation by ranking**, not per-user error — no individual ground-truth treatment effect ever exists to check a prediction against. Decile table: bin held-out users by predicted CATE, report the actual measured ATE per bin. Qini curve / AUUC as a whole-ranking summary.
5. **Policy curve**: incremental conversions vs. fraction targeted, under assumed economics, with the cutoff chosen and measured on independent halves of the eval set (see the caveat on this in Section 5).
6. **The "do it wrong" comparison**: rerun the ATE using `exposure` instead of `treatment`.

### 2.2 Model choice: pros and cons

All CATE models here use the same gradient-boosted base learner (`HistGradientBoostingClassifier`/`Regressor`); what differs is how each combines outcome models with treatment assignment. They were tried in this order specifically because each one fixes a named flaw in the last:

- **S-learner** (one model, `treatment` as just another feature). *Pro*: simplest, cheapest, and — on `visit`, where treatment effect is a large share of what determines the outcome — it was competitive with every other model on full-data AUUC. *Con*: a tree-based model rarely splits on `treatment` when other features explain far more of the outcome variance, so the effect gets shrunk toward zero. This is fatal for rare outcomes: on `conversion`, the S-learner collapsed to predicting only 3 distinct values across ~5.6M held-out users — it never had enough signal-carrying splits on `treatment` to represent heterogeneity at all. Kept in the pipeline as a foil, precisely to make that shrinkage visible.
- **T-learner** (two independently-fit models, one per arm, subtracted). *Pro*: no shrinkage — each model is free to fit its arm without competing against other features for `treatment` splits. *Con*: at an 85/15 split, `model_control` is fit on a much smaller sample (here, ~1.3M vs ~7.1M training rows), and because the T-learner is a raw subtraction of two separately-fit models, that model's extra noise passes straight through into the final CATE estimate with nothing to dampen it. This showed up directly in the full-data AUUC on `conversion` (T-learner clearly weakest of the four non-degenerate models).
- **X-learner** (cross-impute each arm's individual treatment effect using the *other* arm's model, fit new regressors on those imputed effects, blend by propensity). *Pro*: designed specifically for this imbalance — instead of using the noisy `model_control` predictions raw, it launders them through a large-sample regression (`tau1`, fit on the ~7.1M-row treated training group) before they ever reach the final estimate. This directly targets the T-learner's failure mode. *Con*: two extra models to fit and more moving parts (four base models plus a propensity model) for a benefit that's only large when the treatment groups are meaningfully imbalanced — at 50/50 it wouldn't earn its complexity.
- **DR-learner** (doubly robust pseudo-outcome: outcome-model estimate plus a propensity-weighted residual correction, one final smoothing model). *Pro*: consistent if *either* the outcome models or the propensity model is correctly specified, not both — a real hedge against outcome-model misspecification that the X-learner doesn't have. On the full data it was tied for best (with the X-learner) on `conversion` AUUC and top-decile precision. *Con*: needs a working propensity estimate, and it is data-hungry. An early version of this pipeline assumed a constant propensity (~0.85, since treatment was "unconditionally randomized"), which turned out to be wrong once the covariate-balance check found assignment wasn't exactly independent of the features (Section 3). The fix (estimated propensity plus cross-fitted nuisance models) makes it noisier at small n: on the 1M-row sample, with only ~50K controls (about 100 control conversions) per fold, its `conversion` ranking was no better than random (AUUC about -14), whereas on the full data it is among the best. See 2.3.
- **Causal forest** (`econml.dml.CausalForestDML`, honest splitting, double-ML nuisance models). *Pro*: the only model here that targets treatment-effect heterogeneity directly in its splitting criterion, rather than wrapping an outcome-prediction model; "honest" splitting (disjoint subsamples decide splits vs. estimate leaf effects) guards against a specific overfitting failure mode the meta-learners don't have to worry about. *Con*: doesn't scale — fitting it on the full ~14M rows wasn't practical, and fitting it on a subsample wouldn't be a fair comparison against the other four models, which all saw the full training set. Kept in the 1M-row development notebook only, and excluded from the full-data numbers in Section 3 rather than silently approximated.

None of the meta-learners can be validated the way a normal supervised model can: there is no per-user ground-truth treatment effect to check a prediction against (a user is only ever observed treated *or* control, never both), so "which model is better" can only be judged by the decile/Qini ranking check on held-out data — never by a per-prediction error metric.

### 2.3 Experimentation notes

The project developed on a 1M-row sample before scaling to the full ~14M rows, and several methodological problems surfaced only as the pipeline was pushed harder:

- **Propensity assumption caught by the balance check.** The DR- and X-learners originally used a constant propensity (`train["treatment"].mean()` ≈ 0.85), justified by "treatment was unconditionally randomized." The full-data covariate balance check in Section 3 found that assumption isn't quite right — the features predict `treatment` with AUC ≈ 0.51, not exactly 0.5 — so both learners were changed to fit `e(x)` from the features instead of assuming a constant. The DR-learner's outcome models (`mu1`, `mu0`) were also changed to be cross-fit (2-fold), so no row's nuisance prediction comes from a model that was trained on that row — otherwise the residual correction term would be biased toward zero exactly where it's needed most.
- **Policy curve: caught a winner's-curse bug via a split-sample check.** An early version of the policy curve picked "significant" deciles (Bonferroni-corrected CI excludes zero) and used the *same* eval set to measure their profit. That's selection and measurement on identical data — the deciles that pass the significance bar are, by construction, the ones where noise happened to push the estimate up, so crediting them with their own point estimate is optimistic. Splitting the eval set in half (one half selects, the other half measures) directly exposed this. On the 1M-row sample (notebook 05), the set of "significant" deciles was different for every one of five random seeds, and in 9 of the 10 (seed, decile) pairs selected on one half the effect measured on the independent half was lower (for example 0.0048 falling to 0.0030, and 0.0020 to 0.0001). This is the same principle as never evaluating a CATE model on its own training data, one level removed — applied to a selection decision rather than a model fit.
- **Cross-fitting the DR-learner cost precision at small n.** After the DR-learner was changed to estimate its propensity and cross-fit its outcome models, its `conversion` ranking on the 1M-row sample collapsed: AUUC of about -14 (no better than random), and its lowest-ranked decile showed the largest measured effect. Each fold has only ~50K controls (about 100 control conversions), so the doubly-robust pseudo-outcome, whose residuals are weighted by up to 1/(1-e) ≈ 7, is mostly noise. On the full data (~2M controls) the same code is among the best models. The lesson: a method's asymptotic advantage (here, robustness to outcome-model error) is not free at small sample sizes, especially for rare outcomes.
- **Sample-size progression exposed a stability problem, not just a precision one.** At 1M rows, with an earlier version of the DR-learner, several middle deciles looked marginally profitable and a specific "optimal fraction" (70%) seemed to emerge from the Bonferroni-corrected policy curve. At full scale (~559K users per decile instead of ~15K), those same middle deciles resolved to measured effects near or below the cost breakeven, and the split-sample check above showed the "optimal fraction" was never stable across random seeds even at full scale. The lesson carried into Section 5: more data fixed some things (CI width) and clarified that other things (the exact cutoff) were never identified to begin with.

## 3. Results

### Covariate balance

All 12 SMDs are small in absolute terms (largest ~0.05, well under the conventional 0.1 threshold), but at 14M rows that threshold is too lenient: the standard error of an SMD under true randomization is `sqrt(1/n_treated + 1/n_control)` ≈ 0.00075 at this sample size, so an SMD of 0.05 is roughly 65 standard errors from zero — not sampling noise. A gradient-boosted classifier can predict `treatment` from the 12 features with AUC ≈ 0.51 (vs. 0.50 under true independence). So assignment is very close to random, but not exactly independent of the features. This is a small but real effect, not a design flaw the analysis can wave away — see Section 5.

A sample ratio mismatch (SRM) test is a different check and it passes cleanly: the observed split (11,882,655 treated / 2,096,937 control) is within 2 users of the intended 85/15 (chi-square p ≈ 0.999). SRM asks whether the assignment produced the right *counts*; the balance checks above ask whether it was independent of *who the users are*. This dataset passes the first and only mostly passes the second, so a clean SRM does not certify balance.

### ATE and MDE

| outcome | rate, treated | rate, control | ATE | 95% CI | relative lift | MDE (relative) |
|---|---|---|---|---|---|---|
| `visit` | 0.04854 | 0.03820 | 0.01034 | [0.01006, 0.01063] | 27.1% | ~1.1% |
| `conversion` | 0.00309 | 0.00194 | 0.00115 | [0.00108, 0.00122] | 59.5% | ~5.0% |

Both effects are unambiguous — many multiples of their own MDE, CIs nowhere near zero. The full data narrows the `conversion` CI about 4x relative to the 1M-row sample, because the ~2.1M-user control group (not the ~12M treated) is what bounds precision.

### Covariate-adjusted ATE

Because assignment isn't perfectly independent of the features, the raw difference-in-means ATE was checked against three adjusted estimators: a linear regression with treatment × feature interactions, a cross-fitted doubly-robust (DR) estimator with an estimated (not assumed-constant) propensity, and CUPED (pooled-OLS covariate adjustment).

| outcome | raw ATE | linear-adjusted | cross-fitted DR (4M-row subsample) | CUPED |
|---|---|---|---|---|
| `visit` | 0.01034 | 0.00773 | 0.00734 (raw on same rows: 0.01008) | 0.00698 |
| `conversion` | 0.00115 | 0.00100 | 0.00108 (raw on same rows: 0.00114) | 0.00092 |

For `visit`, all three adjustments agree and land ~25-32% below the raw estimate, a gap far too large to be noise (~18 raw standard errors for the linear one). The DR estimate is stable across model seeds (0.00733 to 0.00736). For `conversion`, all three also land below raw (linear about -13%, CUPED about -20%, DR about -5%), but by much less, and the DR estimate moves with the model seed (0.00105 to 0.00109 across five seeds, against a raw of 0.00114), so the size of any `conversion` bias is not pinned down.

CUPED normally reduces variance without moving the point estimate. Here it moved the estimate by ~23 raw standard errors for `visit`, which is the tell that it is acting as a second bias adjustment, not a clean variance-only tool: its "cannot introduce bias" guarantee assumes the covariates are independent of assignment, which the balance check showed is only approximately true. Its variance reduction (25% for `visit`, 11% for `conversion`) is real but secondary here. The honest statement for `visit` is a range: the ATE is somewhere between ~0.0070-0.0077 (adjusted) and ~0.0103 (raw), i.e. roughly a 18-27% relative lift, not a single clean number. Section 5 covers why this range can't be collapsed further with this data.

### CATE and heterogeneity

Heterogeneity is real but concentrated in the top decile of predicted CATE, not spread across the ranking:

- **`visit`**: top decile measured ATE ≈ 0.060-0.065 (~6x the overall 0.0106). Deciles 0-6 are flat, near 0.0003-0.001.
- **`conversion`**: top decile measured ATE ≈ 0.0069-0.0086 for the T-, X- and DR-learners (~6-7x the overall 0.00116). Deciles 1-8 sit near 0.0001-0.0002, several CIs crossing zero.

With ~559K users per decile at full scale, these are precise estimates — the flat middle is evidence of near-zero effect, not evidence obscured by noise.

Model comparison (single train/eval split; AUUC has no attached uncertainty, so read loosely): on `visit` all four models are within ~7% of each other on AUUC. On `conversion` the S-learner collapses (predicts only 3 distinct values — the shrinkage failure mode expected when treatment effect is small relative to what covariates explain for a rare outcome); DR- and X-learner are the best and effectively tied; T-learner is clearly weaker, consistent with the T-learner's core flaw — subtracting two independently-fit models lets the smaller control group's noise pass straight through unfiltered.

**Sleeping dogs**: no decile, in any model, on either outcome, has a CI entirely below zero. There's no evidence ads hurt an identifiable segment of users. This is a statement about deciles, not individuals — a small harmed subgroup could be averaged away inside a decile — and the anonymized features mean even a real one couldn't be described.

### Policy curve

Under assumed economics ($50/conversion, $0.01/impression — not Criteo's real figures), the recommendation depends on the cost model:

- **If cost is billed per eligible user** (every targeted person costs $0.01): the top decile alone captures ~$234K in eval-set profit. Going further changes it by at most ~2% either way (range across all ten cutoffs: ~$228K to ~$235K) — the curve is close to flat beyond the top decile, not still climbing.
- **If cost is billed only when an ad is actually delivered** (`exposure=1`, ~3.6% of eligible users): the middle deciles are exposed only ~2% of the time, so their true cost is ~$0.0002 per eligible user against ~$0.0075 of measured benefit — profitable. Under this model, profit rises monotonically to targeting everyone (~$288K).

Both cost models are consistent with the same measured decile effects; nothing about the users or the effect sizes changes between them. What changes is the denominator being billed — the fraction of eligible users who ever see an ad is ~3.6%, so "pay per eligible user" and "pay per delivered impression" are roughly 28x apart in effective cost per user. Which billing model matches how a real campaign is bought is not something this dataset can answer. The defensible claim: **the top ~10% of users captures most of the value under either cost model; whether it's worth going further is a question about ad-buying mechanics, not about the causal estimate.**

A split-sample check (cutoff chosen on one half of the eval set, profit measured on the independent other half, across 5 seeds) found the exact optimal cutoff unstable — it moved between the top 2 and top 10 deciles depending on the seed, though profit at each chosen cutoff was always within a few percent of that half's own best. An earlier development-sample result claiming a specific optimal fraction (0.7) does not hold up and is not part of this write-up's conclusion.

## 4. Results — the `exposure` vs. `treatment` comparison ("doing it wrong")

| outcome | correct ATE (`treatment`) | naive ATE (`exposure`) | bias multiple |
|---|---|---|---|
| `visit` | 0.01081 (28.6% relative lift) | 0.37918 (~1071% relative lift) | 35.1x |
| `conversion` | 0.00111 (55.3% relative lift) | 0.05308 (~3976% relative lift) | 47.8x |

(Figures from the 1M-row development sample — note the correct-ATE column here (0.01081/0.00111) differs slightly from the full-data ATE in Section 3 (0.01034/0.00115); both are valid intent-to-treat estimates, just on different-sized draws. The mechanism and magnitude of the exposure-bias are not sensitive to sample size, so this comparison wasn't rerun on the full data.) Conditioning on `exposure` compares the ~3.6% of eligible users the auction chose to show ads to — selected specifically because the algorithm predicted they'd convert — against everyone else. That's comparing an algorithmically-selected high-propensity slice to a general population, not two comparable groups. It is not a subtle bias: relative lift figures above ~1000% should be a red flag on their own, independent of knowing to check whether the grouping variable was randomized.

This is the sharpest illustration in the whole analysis of why the identification argument in Section 1 matters: it's a concrete, quantified demonstration of exactly the mistake randomization was designed to prevent.

## 5. Why these numbers might be wrong

- **Covariate imbalance is small but not zero, and its source is unknown.** The features predict `treatment` with AUC ≈ 0.51 — weak, but not exactly 0.5. For `visit`, three different adjustments (regression, doubly-robust, CUPED) all move the ATE by ~25-32%; for `conversion` all three move it in the same direction but by only ~5-20%, and that size is not pinned down. Because the features are anonymized, the mechanism can't be diagnosed from the data. Three candidate explanations, none confirmed: (1) Criteo's non-uniform privacy subsampling of the released dataset could itself be correlated with arm and outcome, even if the original assignment was clean; (2) if any feature was measured post-assignment rather than pre-treatment, adjusting for it would condition on a post-treatment variable — the same error as Section 4, one level more subtle — and could make the estimate *more* biased, not less; (3) stratified or non-user-level randomization. This means the `visit` ATE should be reported as a range (~18-27% relative lift), not a point estimate, and the `conversion` ATE, while less affected, isn't fully cleared either.
- **The policy recommendation depends on an unverifiable assumption about ad-buying mechanics** (cost per eligible user vs. cost per delivered impression) — a ~28x swing in effective cost per user, driven entirely by billing structure, not by anything causal. The two cost models produce different "optimal" answers (top-decile-only vs. target-everyone) from the identical underlying effects.
- **The exact policy cutoff is not identified.** A split-sample check shows the "optimal fraction" is unstable to sampling — anything claiming a specific cutoff percentage (e.g. "70%") is overfit to a single data split and shouldn't be trusted as a fixed number.
- **Absolute effect sizes are not Criteo's real business numbers.** The dataset was non-uniformly subsampled for privacy. Relative comparisons (lift %, bias multiples, decile ratios) are more trustworthy than the absolute rates.
- **No user-level interpretation is possible.** Features are anonymized and randomly projected — the analysis can rank users by predicted CATE but cannot say what kind of user is in the top decile. Any resume bullet or business narrative built on "these are high-value users because X" is not supported by this data.
- **Dollar figures in the policy curve are illustrative**, built on assumed ($50/conversion, $0.01/impression), not real, economics.
- **No individual-level validation is possible for CATE**, by construction — no per-user ground-truth treatment effect exists (a user is only ever observed treated or control, never both). All validation here is at the ranking/decile level on held-out data, which is a weaker but the only available check.

## 6. What's not in this analysis

- Causal forest results are from the 1M-row development sample only (`notebooks/03_cate_estimation.ipynb`); `econml`'s `CausalForestDML` does not scale to the full ~14M rows and was excluded from the full-data run rather than approximated on a subsample that wouldn't be comparable to the other models.
