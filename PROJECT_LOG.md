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

## Next Steps

1. Vectorize CPT
2. Bayesian stacking (Dirichlet posterior over weights)
3. Add MOT (Mixture of Theories)
4. Della HPC for Bayesian runs
5. Paper figures
