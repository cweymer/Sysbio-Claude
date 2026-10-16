# Changelog

## 2026-10-16 - initial analysis

- Added `project_4_analysis.ipynb`, an eight-section regression analysis of
  `project_4_dataset.csv`, executed with outputs.
- Added `data_tools.py` (loading and validation, distribution summaries, Pearson/Spearman
  correlation table, Benjamini-Hochberg adjustment).
- Added `modeling.py` (OLS and WLS fits, VIF, per-predictor curvature diagnostics, pairwise
  interaction scan, Ramsey RESET, Breusch-Pagan, log-squared-residual variance model,
  Freedman-Lane permutation test, pairs bootstrap with the variance model re-estimated in each
  resample). Seeded at `SEED = 20261016`.
- Added `analysis_plots.py` with seven figures in the Okabe-Ito palette, Arial text and
  editable SVG output, written to `figures/` as SVG and PNG.
- Findings: an `x1`-by-`x2` product term and an `x3` quadratic raise R-squared from 0.085 to
  0.908; error variance increases multiplicatively with `x4`; residual non-normality in the
  unweighted fit is explained entirely by that variance function.
- `breusch_pagan_p` takes an optional explicit residual vector so the test can be applied to
  the standardised residuals of a weighted fit rather than to its raw-scale residuals.
- The auxiliary variance regression applies the -E[log chi^2_1] = 1.2704 bias correction to its
  intercept (`modeling.LOG_CHI2_BIAS`), so the quoted sigma(x4) and the standardised residuals
  are on the right scale; the weighted coefficient estimates are unaffected, since they are
  invariant to a common rescaling of the weights.
- `permutation_test_term` resolves formula terms to patsy parameter names, so `I(x3**2)` and
  `I(x3 ** 2)` both work.
- `residual_diagnostics` returns NaN for a predictor whose quadratic term is already in the
  model instead of fitting a rank-deficient augmentation.
