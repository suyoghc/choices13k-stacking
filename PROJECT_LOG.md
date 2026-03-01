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

## Next Steps

1. ~~Vectorize CPT~~ ✓
2. ~~Bayesian stacking~~ ✓
3. ~~Hierarchical Bayesian stacking~~ ✓
4. Add MOT (Mixture of Theories)
5. Della HPC for full runs
6. Paper figures
