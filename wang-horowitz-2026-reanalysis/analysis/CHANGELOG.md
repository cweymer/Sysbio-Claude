# Changelog

All notable changes to the analysis code.

## [0.1.0] - 2026-10-14

### Added
- `stats_helpers.py`: dog-level aggregation, sign-flip and label permutation tests,
  Hodges-Lehmann shift with percentile bootstrap, matched-pairs rank-biserial,
  epsilon-squared, Benjamini-Hochberg adjustment, ICC(2,1), Cohen's kappa, and
  `min_detectable_d` (noncentral-t sensitivity).
- `stats_helpers.h_max`: maximum attainable Kruskal-Wallis H for given group sizes,
  used to establish that the published Part 2 tests were run on time bins rather
  than on dogs.
- `models.py`: Part 1 mixed model (dog random intercept, period covariate), beta
  regression on the dual-box preference index, binomial GLMM on bin-level binary
  data with a dog random effect, multinomial model for the compositional posture
  triple.
- `simulate.py`: synthetic fixtures matching the documented deposit structure, so the
  pipeline is runnable and testable before the Dryad deposit is accessible.
- `wang_horowitz_reanalysis.ipynb`: the analysis in order — distributions, Part 1
  dog-level contrasts and mixed model, the H-ceiling diagnostic, Part 2 dog-level
  tests and GLMM, sensitivity, coder reliability, and a single auditable test register.
- `tests/test_helpers.py`: validates `h_max` against the 9/9/6 and 12/12/12 ceilings
  and `min_detectable_d` against the review's published sensitivity figures.

### Notes
- `min_detectable_d` was changed from a normal approximation to a noncentral-t solve
  during development; the normal version was optimistic by ~0.09 d at n = 9 per group.
- `fit_part2_binomial_glmm` takes bin-level binary rows rather than aggregated
  successes/failures, because `BinomialBayesMixedGLM` requires a 0/1 endog.
