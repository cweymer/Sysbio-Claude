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

## 2026-10-16 - numeric corrections to the narrative

- The heteroscedasticity claim now separates the binned empirical residual SD (0.62 to 2.65
  across sixths of `x4`, which is what figure 5 shows) from the fitted variance function
  (0.58 at `x4` = 0 rising to 3.48 at `x4` = 3). The earlier text quoted the binned range while
  describing it as the model's range, and gave the information ratio as a sixteenth rather
  than the 4.3-fold SD ratio's eighteenth.
- Shapiro-Wilk figures corrected: the additive model's residuals give p = 2.2e-16 (skew 1.08,
  excess kurtosis 4.46); the p = 5.4e-7 previously attributed to "raw residuals" is the
  unweighted structured model's, now labelled as such.
