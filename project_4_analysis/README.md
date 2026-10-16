# Project 4 dataset: regression analysis

A worked regression analysis of `project_4_dataset.csv` (500 rows; four continuous predictors
`x1`-`x4` and a continuous response `y`). The dataset contains several deliberate hazards, and
the analysis is organised so that each one is found by a diagnostic rather than assumed.

## What the analysis concludes

The fitted model is

```
y = -3.00 + 3.00 * x1 * x2 + 2.97 * x3^2 + 1.50 * x4 + e,     sd(e) = exp(-0.54 + 0.60 * x4)
```

R-squared 0.908, against 0.085 for the additive model `y ~ x1 + x2 + x3 + x4`.

| Hazard | Present? | Evidence in the notebook |
|---|---|---|
| Collinearity | No | All VIFs 1.0; largest predictor-predictor correlation 0.059 |
| Pure interaction `x1` x `x2` | Yes | Marginal r = 0.018 and 0.080; adds 0.318 to R-squared; t = 69 |
| Symmetric nonlinearity in `x3` | Yes | Marginal r = -0.096; linear coefficient p = 0.29; quadratic t = 72 |
| Heteroscedasticity in `x4` | Yes | Breusch-Pagan p = 3.9e-14; residual SD runs 0.6 to 2.6 across the `x4` range |
| Non-normal errors | No | Raw residuals fail Shapiro-Wilk (p = 5.4e-7); standardised residuals pass (p = 0.95) |

The last row is the point of the exercise: the heavy-tailed, skewed residuals of the naive fit
are produced by the variance function, not by a non-normal error family, so the correct fix is
a variance model rather than a transform of `y` or a robust loss.

## Files

```
project_4_analysis/
  project_4_analysis.ipynb   entry point: the narrative, executed with outputs
  data_tools.py              loading, distribution summaries, correlations, BH adjustment
  modeling.py                OLS/WLS fits, VIF, curvature and interaction scans, RESET,
                             Breusch-Pagan, variance model, permutation test, bootstrap
  analysis_plots.py          all seven figures (Okabe-Ito palette, Arial, SVG + PNG)
  project_4_dataset.csv      the data as supplied
  figures/                   fig1 ... fig7, each as .svg (editable) and .png
  CHANGELOG.md
  README.md
```

## Running it

Requires `python >= 3.10` with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`,
`jupyter`. From this directory:

```bash
PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace project_4_analysis.ipynb
```

`PYTHONPATH=.` is needed because the modules sit beside the notebook rather than being
installed. The permutation test and bootstrap are seeded (`modeling.SEED = 20261016`), so the
numbers above reproduce exactly.

## Caveats

- Model selection and inference use the same 500 rows, so p-values for the *selected* terms are
  optimistic in the usual post-selection sense. The two retained structural terms have |t| > 69,
  far from any decision boundary, but a borderline term would need a held-out check.
- The exponential variance function is a description of these residuals, not a mechanism. The
  pairs-bootstrap intervals in the notebook do not depend on that form and should be quoted if
  it is in doubt.
- `x1`-`x3` span roughly -3 to 3 and `x4` is confined to 0 to 3. The quadratic and product terms
  make extrapolation outside those ranges unreliable.
