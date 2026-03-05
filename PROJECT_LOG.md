# Project Log — choices13k Stacking

## Session 1: Planning (2026-02-28)

**Context established:**
- choices13k = 13,006 risky choice problems, ~25 participants each (Peterson et al., Science 2021)
- Goal: Bayesian stacking to ask "which decision theory wins where?"
- Inspired by Almaatouq/Griffiths integrative experiment design framework
- Peterson's HURD library exists but has stale JAX dependency — decided to reimplement models

**Key design decisions:**
- Follow Poldrack's "Better Code, Better Science" heuristics throughout
- Lightweight implementations of EV/EU/PT/CPT (avoid HURD dependency)
- Standard project layout: src/stacking/, tests/, data/, scripts/

## Session 2: Scaffolding + MVP (2026-03-01)

**Built:**
- Config dataclasses (no magic numbers, no long param lists)
- Data loader with defensive assertions (bRate ∈ [0,1], positive sample sizes, etc.)
- EV, EU, PT, CPT model classes (predict only — fitting is separate module)
- MLE fitting via scipy L-BFGS-B with multi-start
- K-fold stacking pipeline (Breiman 1996)
- Hierarchical stacking with intercept + feature regression
- 31 tests — all passing

**Data facts discovered:**
- 14,568 rows (not 13,006 — some problems have both feedback conditions)
- Problem JSON keyed "0"–"14567" (string index ≠ Problem column in CSV)
- bRate mean=0.52, well-centered

**Results — uniform stacking (5-fold, EV/EU/PT):**
- EV: MSE=0.0297, weight=0.049
- EU: MSE=0.0263, weight=0.000
- PT: MSE=0.0234, weight=0.951
- Stacked MSE: 0.0234 (0.09% improvement over best single)

**Results — hierarchical stacking:**
- Hierarchical MSE: 0.0233 (0.32% over uniform)
- Baseline weights (intercept): EV=4%, EU=10%, PT=86%
- EU weight ranges 5%–19% across problems

**Key finding — what shifts EU weight up:**
- More lottery outcomes (β=+0.094) — complex gambles
- Larger max payoffs (β=+0.086) — high stakes
- Wider outcome range (β=+0.066)
- Interpretation: probability weighting hurts when there are many probabilities to distort

**Key finding — what locks in PT:**
- Ambiguity (β=−0.100) — strongest single effect
- When probabilities themselves are uncertain, PT's heuristic distortions matter most

**Bottleneck identified:**
- CPT loops per-problem in Python — too slow for full runs
- Needs vectorization before 4-model stacking is feasible

## Session 3: CPT Vectorization (2026-02-28)

**Problem solved:**
- CPT was looping per-problem in Python (925ms for 14,568 problems)
- Blocked 4-model stacking and Bayesian runs

**Solution:**
- Fully vectorized CPT using numpy array ops only
- Key insight: after sorting outcomes ascending (worst→best), losses cluster left, gains cluster right
- Losses: forward cumsum, apply weighting, take differences for decision weights
- Gains: reverse cumsum (flip, cumsum, flip back), same weighting logic
- Masking handles variable gain/loss mix per problem

**Performance:**
- Loop-based: 925ms
- Vectorized: 27ms
- **35x speedup**, numerically identical (max diff ~10⁻¹⁵)

**Results — 4-model stacking (5-fold, EV/EU/PT/CPT):**
| Model | MSE | Weight |
|-------|-----|--------|
| EV | 0.0297 | 0% |
| EU | 0.0263 | 0% |
| PT | 0.0234 | 44.9% |
| CPT | 0.0230 | **55.1%** |
| **Stacked** | **0.0224** | — |

**Key finding:**
- CPT beats PT when included in the race
- Previous PT dominance (95%) was artifact of CPT's absence
- CPT gets majority weight (55%) — rank-dependent probability weighting matters
- Stacked ensemble improves 4.4% over best single model

**Tests:**
- Added 5 CPT-specific tests (batch consistency, loss aversion, mixed gains/losses)
- All 36 tests passing

## Session 4: Bayesian Stacking (2026-03-01)

**Built:**
- New module `src/stacking/bayesian.py` with PyMC implementation
- Dirichlet prior on weights (uniform, α=1)
- Binomial likelihood using sample sizes from `n` column
- 8 new tests — all passing (44 total)

**Model:**
```
weights ~ Dirichlet(1, 1, 1, 1)
mu = oof_predictions @ weights
y ~ Binomial(n, mu)
```

**Results — Bayesian stacking (4 chains × 2000 draws):**
| Model | Mean | Std | 94% HDI |
|-------|------|-----|---------|
| EV | 1.1% | 0.8% | [0.0%, 2.4%] |
| EU | 0.2% | 0.2% | [0.0%, 0.6%] |
| PT | 41.4% | 1.7% | [38.1%, 44.5%] |
| **CPT** | **57.3%** | **1.7%** | **[54.1%, 60.3%]** |

**Diagnostics:**
- R-hat: 1.0000 (perfect convergence)
- Min ESS: 1525 (well above 400 threshold)

**Key finding:**
- CPT's 94% HDI excludes 0.5 — we're >97% confident CPT gets majority weight
- This is a **statistically significant** result, not just a point estimate
- The uncertainty is tight (±1.7%) — 14,568 problems give precise posteriors

**Comparison to frequentist:**
- Frequentist: CPT 55.1%, PT 44.9%
- Bayesian: CPT 57.3% [54.1%, 60.3%], PT 41.4% [38.1%, 44.5%]
- Slight difference due to Binomial likelihood vs MSE loss

## Session 5: Hierarchical Bayesian Stacking (2026-03-01)

**Built:**
- Extended `bayesian.py` with hierarchical model
- Weights vary by problem features: `w_k(x) = softmax(intercept + X @ beta)`
- Hierarchical shrinkage prior on beta: `sigma_beta ~ HalfNormal(1)`, `beta ~ Normal(0, sigma_beta)`
- 4 new tests (12 total Bayesian tests), 48 tests total

**Model:**
```
sigma_beta ~ HalfNormal(1)
beta ~ Normal(0, sigma_beta)    # (n_features, n_models-1)
intercept ~ Normal(0, 2)        # (n_models-1)
logits = intercept + X @ beta   # last model is reference
weights = softmax(logits)
mu = sum(oof_predictions * weights)
y ~ Binomial(n, mu)
```

**Results — hierarchical Bayesian (features: LotNumB, Amb, Corr):**

Baseline weights (no features):
| Model | Mean | Std |
|-------|------|-----|
| EV | 0.5% | 0.4% |
| EU | 6.1% | 1.6% |
| PT | 26.4% | 3.5% |
| CPT | 67.1% | 3.0% |

Feature effects (* = credibly non-zero):
| Feature | Effect |
|---------|--------|
| LotNumB | PT +0.70*, EU +0.42 (more outcomes → weighting helps) |
| Amb | EU -1.84* (ambiguity kills rational EU) |
| Corr | PT +0.16* (correlation favors PT) |

Weight variation across 14,568 problems:
- CPT: mean=62.6%, range=[18.7%, 85.3%]
- PT: mean=26.5%, range=[9.1%, 81.2%] — huge variation!
- EU: mean=10.6%, range=[0.1%, 29.4%]

**Key findings:**
- CPT dominates on average but PT can reach 81% for some problems
- Ambiguity strongly penalizes EU (rational model fails under uncertainty)
- More lottery outcomes shift weight toward probability-weighting models (PT/CPT)

**Diagnostics:** R-hat=1.00, ESS=3152 (excellent)

## Session 6: Mixture of Theories (2026-03-01)

**Built:**
- MOT model in `bayesian.py` with Beta-Binomial overdispersion
- Interprets weights as "proportion of participants using each theory"
- 4 new tests (52 total)

**Model:**
```
π ~ Dirichlet(1, 1, 1, 1)   # mixture proportions
κ ~ HalfNormal(10)           # overdispersion
μ = oof_predictions @ π      # expected P(B) under mixture
y ~ BetaBinomial(n, μ*κ, (1-μ)*κ)
```

**Results:**
| Theory | Usage % | 94% HDI |
|--------|---------|---------|
| EV | 1.4% | [0.0%, 3.1%] |
| EU | 0.4% | [0.0%, 1.0%] |
| PT | 40.0% | [35.5%, 44.0%] |
| **CPT** | **58.3%** | **[54.2%, 62.5%]** |

**Overdispersion:** κ = 21.9 ± 0.6 (fairly high = mixture explains most variance)

**Interpretation:**
- 58% of participants use CPT as their decision rule
- 40% use PT
- Almost no one uses EU or EV (<2% combined)
- The mixture model captures individual heterogeneity well (high κ)

**Diagnostics:** R-hat = 1.00, ESS = 2176

## Session 7: Production Results & Comparison (2026-03-01)

**Della HPC run completed:**
- 10-fold CV, 4 chains × 4000 samples, 2000 tune
- Total runtime: 35.3 minutes
- All diagnostics passed (R-hat = 1.00, ESS > 2000)

**Production results:**
| Model | Frequentist | Bayesian Mean | 94% HDI | MOT Usage |
|-------|-------------|---------------|---------|-----------|
| EV | 0.0% | 1.1% | [0.0, 2.4] | 1.4% |
| EU | 0.0% | 0.2% | [0.0, 0.6] | 0.3% |
| PT | 44.9% | 41.5% | [38.1, 44.6] | 40.0% |
| **CPT** | **55.1%** | **57.2%** | **[54.1, 60.5]** | **58.3%** |

MOT overdispersion: κ = 21.9 (high = mixture captures heterogeneity well)

**Comparison with Peterson et al. (2021) Science:**

| Aspect | Peterson 2021 | Our analysis |
|--------|---------------|--------------|
| Question | "Which model predicts best?" | "How much does each contribute?" |
| Method | ML prediction competition | Bayesian model stacking |
| Best accuracy | 84.8% (context-dependent NN) | N/A (stacking, not classification) |
| Classical theories | "Low predictive accuracy" | CPT > PT; both contribute |
| Key insight | Context matters | Among classical, CPT wins |

**Reconciliation:**
- Peterson et al. showed no single classical theory is sufficient — agreed (our ensemble beats any single model by 4.4%)
- They found context-dependent models beat value-based — we didn't test this
- We add: *among* classical theories, CPT > PT with tight uncertainty
- Both agree: EU and EV are empirically dead (<2% combined)

**Key contribution:**
Peterson asked "can we do better than classical theories?" (yes, with ML)
We ask "if forced to weight classical theories, which wins?" (CPT, significantly)

**Files created:**
- `scripts/plot_saved_results.py` — generates figures from saved NetCDF posteriors
- Fixed arviz trace plot compatibility (let arviz create own figures)

**Figures generated:**
- `fig1_comparison.pdf` — Frequentist vs Bayesian weights
- `fig2_posteriors.pdf` — Posterior distributions
- `fig3_hierarchical.pdf` — Feature effects on weights
- `fig4_mot.pdf` — MOT proportions + overdispersion
- `fig5a_bayes_diagnostics.pdf`, `fig5b_mot_diagnostics.pdf` — MCMC traces

## Session 8: Code Review & Fixes (2026-03-01)

**Reviewed full codebase** (built by Opus 4.5 in sessions 1–7). Found and fixed 6 issues:

1. **Bug: `compute_stacking_weights` ignored `loss` parameter** — always used MSE even when `"cross_entropy"` was passed. Wired up `loss_fn` dispatch.
2. **Weak alignment check in `prepare_gamble_data`** — used `or` (either outcome or probability matches), which could pass by coincidence on misaligned data. Replaced with order-independent EV comparison (`json_ev_a` vs `csv_ev_a`).
3. **`assert` for data validation** — all data checks in `data.py` and `bayesian.py` used `assert`, which is silently disabled by `python -O`. Replaced with `if/raise ValueError` throughout.
4. **`_validate_problems` only checked first problem** — now loops through all 14,568 problems.
5. **Unused standalone functions** — `compute_expected_value`, `apply_power_utility`, `apply_probability_weight` duplicated logic already in model classes. Removed along with their 6 tests (properties already covered by model class tests).
6. **`sys.path.insert` hack in tests** — redundant with `pip install -e`. Removed.

**Tests:** 46 passing (was 52; removed 6 tests for deleted functions).

**Commit:** `960e6ab` — pushed to main.

## Session 9: Context-Dependent Model (2026-03-02)

**Motivation:** Peterson et al. (2021) showed context-dependent models (84.8% accuracy) far exceeded classical theories. Classical theories compute V(gamble) independently, but human decisions depend on the *relationship* between the two gambles in the choice set. Adding a context-dependent model answers: "how much do classical theories contribute beyond a data-driven context model?"

**Built:**
- Extended `GambleData` with optional `features` field (backward compatible)
- `_build_context_features()` computes 16 features:
  - Raw gamble features (6): Ha, pHa, La, Hb, pHb, Lb
  - Cross-gamble context features (6): ev_diff, max/min outcome diffs, prob_asymmetry, outcome ranges
  - Design variables (4): feedback, ambiguity, correlation, LotNumB
- `ContextModel` class wrapping sklearn `GradientBoostingRegressor` (200 trees, depth 4, lr 0.1, subsample 0.8)
- Pipeline dispatch via `is_sklearn_model` flag — sklearn models use `fit()`/`predict()` instead of scipy MLE
- Updated `run_bayesian_full.py` to include all 5 models; fixed hardcoded index in summary output
- 4 new tests (50 total), all passing

**Draft methods/results section** written to `draft_methods_results.md` (gitignored).

**Commits:** `049f809`, `7282e58` — pushed to main.

**Next:** Run 5-model stacking + Bayesian analyses on Della.

## Session 10: 5-Model Production Results (2026-03-02)

**Della HPC run completed:**
- 5 models (EV, EU, PT, CPT, Context), 10-fold CV, 4 chains × 4000 samples, 2000 tune
- Total runtime: 35.7 minutes
- All diagnostics passed (R-hat = 1.00, ESS > 900 across all models)

**Production results — 5-model stacking:**

| Model | Freq. Weight | Bayesian Mean | 94% HDI | MOT Usage |
|-------|-------------|---------------|---------|-----------|
| EV | 0.0% | 0.3% | [0.0, 1.1] | 0.3% |
| EU | 0.0% | 0.4% | [0.0, 1.1] | 0.4% |
| PT | 0.0% | 0.9% | [0.0, 2.2] | 0.9% |
| CPT | 0.1% | 1.5% | [0.0, 2.9] | 1.5% |
| **Context** | **99.9%** | **96.8%** | **[95.7, 98.1]** | **96.7%** |

**OOF MSE:** Context ~0.010 vs CPT ~0.023 (57% better)

**Hierarchical Bayesian (5-model):**
- Context baseline weight: 97.0%
- Feature effects minimal — Context already captures cross-gamble structure
- ESS=904, R-hat=1.00

**MOT overdispersion:** κ = 143.5 (up from 21.9 in 4-model)
- Much tighter fit — Context explains far more variance than classical theories alone

**Key findings:**

1. **Context model crushes classical theories** — 57% MSE reduction over CPT. Cross-gamble features (EV differences, outcome asymmetries, probability contrasts) capture what classical value functions miss.

2. **Classical theories contribute almost nothing beyond context** — CPT drops from 57% (4-model) to 1.5% (5-model). The features that made CPT valuable are subsumed by the GBR's cross-gamble features.

3. **Two-level story for the paper:**
   - Level 1 (classical only): CPT > PT >> EU ≈ EV. Rank-dependent probability weighting wins among classical theories.
   - Level 2 (with context): Context >> all classical combined. Decisions depend on gamble *relationships*, not independent values — confirming Peterson et al.'s core finding with Bayesian uncertainty.

4. **κ jumped 21.9 → 143.5** — the mixture model fits much tighter with Context, less residual heterogeneity.

5. **94% HDI for Context [95.7%, 98.1%]** excludes 95%, meaning we're highly confident Context gets >95% weight.

**Reconciliation with Peterson et al. (2021):**
- Peterson: context-dependent NN gets 84.8% accuracy, far exceeding classical theories
- Our analysis: Context gets 96.8% of Bayesian stacking weight [95.7%, 98.1%]
- Both confirm the same core insight: decisions are context-dependent, not value-based
- We add Bayesian uncertainty quantification to their finding

**Updated draft_methods_results.md** with 5-model results (two-level narrative).

## Session 11: Figures & Paper Draft (2026-03-02)

**Two-level figures generated:**
- Rewrote `scripts/plot_saved_results.py` for two-level narrative
- Loads both 4-model and 5-model results (renamed: `*_4model.nc`, `*_5model.nc`)
- 6 new figures:
  - `fig1_classical_weights` — Frequentist + Bayesian bar charts (CPT 57%, PT 42%)
  - `fig2_classical_posteriors` — KDE posteriors (Bayesian + MOT, classical only)
  - `fig3_hierarchical` — Feature effects forest plot (classical only)
  - `fig4_context_dominance` — 5-model weights with zoomed classical panel
  - `fig5_mot_comparison` — 4-model vs 5-model MOT + kappa shift (21.9 → 143.5)
  - `fig6a/b_diagnostics` — MCMC traces for 5-model runs

**Complete paper draft written** (`draft_methods_results.md`, gitignored):
- Added Introduction (~6 paragraphs):
  - Classical theory hierarchy and empirical landscape
  - Peterson et al. (2021) and choices13k
  - The gap: no uncertainty quantification among classical theories
  - Bayesian stacking as methodological solution
  - Two-level contribution preview
- Added Discussion (~5 subsections):
  - CPT > PT: rank-dependent weighting empirically validated at scale
  - Context-dependence as failure of independent valuation, not utility functions
  - Kappa story (21.9 → 143.5): classical mixture misses context
  - 6 limitations (stacking ≠ cognition, aggregate-level, incomplete theory set, black-box context model, post-hoc features, no individual differences)
  - 4 future directions (SHAP, individual-level stacking, other datasets, interpretable context models)
- Total: ~4,500 words, 337 lines

**Commits:** `6b27a32` — figures + result files + plotting script pushed to main.

## Next Steps

1. ~~Vectorize CPT~~ ✓
2. ~~Bayesian stacking~~ ✓
3. ~~Hierarchical Bayesian stacking~~ ✓
4. ~~MOT (Mixture of Theories)~~ ✓
5. ~~Della HPC for full runs~~ ✓
6. ~~Paper figures~~ ✓
7. ~~Draft methods/results~~ ✓
8. ~~Add context-dependent model~~ ✓
9. ~~Run 5-model analysis on Della~~ ✓
10. ~~Update methods/results with 5-model results~~ ✓
11. ~~Update figures for 5-model results~~ ✓
12. ~~Write Introduction and Discussion~~ ✓
13. Feature importance / SHAP on Context model
14. Clean up repo for sharing (README, dependency pinning)
15. Robustness checks (GBR hyperparams, feedback split, leave-one-feature-out)
