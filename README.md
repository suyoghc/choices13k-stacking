# choices13k-stacking

Bayesian model stacking on the [choices13k](https://github.com/jcpeterson/choices13k) dataset (Peterson et al., *Science* 2021).

**Question:** Which theory of risky choice wins — and where?

## What this does

1. Fits classic decision theories (EV, EU, PT, CPT) to 14,568 risky choice problems
2. Computes K-fold stacking weights (Breiman 1996)
3. Fits hierarchical stacking weights — which model wins for which problem types

## Results so far

- **PT dominates** — 95% uniform stacking weight
- **EU gains ground** for many-outcome, high-stakes problems
- **Ambiguity locks in PT** — strongest predictor of weight variation
- Hierarchical improvement is small (0.32%) — PT dominance is genuine

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## TODO

- [ ] Vectorize CPT
- [ ] Bayesian stacking (PyMC/Stan)
- [ ] Cross-validate hierarchical regularization
- [ ] Della HPC integration
- [ ] Add MOT (Mixture of Theories)

## References

- Peterson et al. (2021). *Science*, 372(6547), 1209–1214.
- Yao et al. (2018). *Bayesian Analysis*, 13(3), 917–1007.
- Poldrack (2025). Better Code, Better Science.
