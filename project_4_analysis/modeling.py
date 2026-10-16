"""Model fitting, diagnostics and inference for the project 4 dataset.

The workflow this module supports is: fit the additive linear model, read the
residuals for curvature and for non-constant spread, add the structure the
residuals demand, then re-check.  Standard statsmodels routines are used
throughout; nothing here reimplements an estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

ADDITIVE_FORMULA = "y ~ x1 + x2 + x3 + x4"
STRUCTURED_FORMULA = "y ~ x1 + x2 + x1:x2 + x3 + I(x3**2) + x4"

SEED = 20261016


def fit_ols(df: pd.DataFrame, formula: str = ADDITIVE_FORMULA):
    """Ordinary least squares fit of `formula`."""
    return smf.ols(formula, data=df).fit()


def vif_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance inflation factors for `columns`, with an intercept included.

    VIF near 1 means the predictor is essentially orthogonal to the others, so
    its coefficient is well identified; values above roughly 5-10 indicate the
    coefficient is poorly separated from its collinear partners.
    """
    X = sm.add_constant(df[columns])
    return pd.DataFrame(
        {
            "predictor": X.columns,
            "vif": [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])],
        }
    ).set_index("predictor")


def residual_diagnostics(fit, df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Per-predictor curvature and variance diagnostics for a fitted model.

    `curvature_p` is the p-value for adding a quadratic term in that predictor
    to the fitted model (an F-test of one restriction).  `abs_resid_rho` is the
    Spearman correlation between the predictor and the absolute residual: a
    positive value means the spread of the residuals grows with the predictor.
    """
    resid = pd.Series(np.asarray(fit.resid), index=df.index)
    rows = []
    for col in predictors:
        if f"I({col}**2)" in fit.model.formula.replace(" ", ""):
            curv_p = np.nan  # quadratic term already in the model
        else:
            aug = smf.ols(f"{fit.model.formula} + I({col}**2)", data=df).fit()
            curv_p = float(aug.compare_f_test(fit)[1])
        rho = stats.spearmanr(df[col], resid.abs())
        rows.append(
            {
                "predictor": col,
                "curvature_p": curv_p,
                "abs_resid_rho": rho.statistic,
                "abs_resid_p": rho.pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("predictor")


def interaction_scan(df: pd.DataFrame, base_formula: str, predictors: list[str]) -> pd.DataFrame:
    """F-test for each pairwise product term added to `base_formula`.

    Returned p-values are raw; apply an FDR adjustment before reading them as a
    set, since one test is run per pair.
    """
    base = smf.ols(base_formula, data=df).fit()
    rows = []
    for i, a in enumerate(predictors):
        for b in predictors[i + 1 :]:
            aug = smf.ols(f"{base_formula} + {a}:{b}", data=df).fit()
            rows.append(
                {
                    "term": f"{a}:{b}",
                    "delta_r2": aug.rsquared - base.rsquared,
                    "p": float(aug.compare_f_test(base)[1]),
                }
            )
    return pd.DataFrame(rows).set_index("term")


def reset_test(fit) -> float:
    """Ramsey RESET p-value: evidence of unmodelled nonlinearity in the mean."""
    return float(linear_reset(fit, power=2, test_type="fitted", use_f=True).pvalue)


def breusch_pagan_p(fit, resid: np.ndarray | None = None) -> float:
    """Breusch-Pagan p-value for non-constant residual variance.

    For a weighted fit pass the standardised residuals explicitly: `fit.resid`
    is on the original scale, where the variance is non-constant by design.
    """
    r = fit.resid if resid is None else resid
    return float(het_breuschpagan(np.asarray(r), fit.model.exog)[1])


# E[log(chi^2_1)] = digamma(1/2) + log 2 = -1.2704; log r^2 is biased low by this amount,
# so the intercept of the auxiliary regression needs it added back before the fitted values
# are read as log variances.  The bias is a constant, so it does not change the WLS estimates
# (which are invariant to a common rescaling of the weights) - only the scale of the
# standardised residuals and of any sigma(x) that is quoted.
LOG_CHI2_BIAS = 1.2704


def variance_model(fit, df: pd.DataFrame, variance_predictor: str = "x4"):
    """Regress log squared residuals on `variance_predictor`.

    This estimates a multiplicative variance function, sigma^2(x) = exp(a + b*x).
    Returns the auxiliary fit; pass it to `variance_weights` to get WLS weights.
    """
    aux_df = df.assign(_log_sq_resid=np.log(np.asarray(fit.resid) ** 2))
    return smf.ols(f"_log_sq_resid ~ {variance_predictor}", data=aux_df).fit()


def log_variance(aux) -> np.ndarray:
    """Bias-corrected fitted log variance from an auxiliary variance regression."""
    return np.asarray(aux.fittedvalues) + LOG_CHI2_BIAS


def variance_weights(aux) -> np.ndarray:
    """WLS weights, 1 / estimated variance, from an auxiliary variance regression."""
    return 1.0 / np.exp(log_variance(aux))


def fit_wls(df: pd.DataFrame, formula: str, weights: np.ndarray):
    """Weighted least squares fit with the supplied weights (1 / estimated variance)."""
    return smf.wls(formula, data=df, weights=weights).fit()


def standardised_residuals(fit, weights: np.ndarray) -> np.ndarray:
    """Residuals rescaled by the square root of the weights.

    If the variance model is right these are homoscedastic, so their histogram
    and Q-Q plot are the ones that should be read for the error distribution.
    """
    return np.asarray(fit.resid) * np.sqrt(weights)


def coefficient_table(fit) -> pd.DataFrame:
    """Estimates, standard errors, t statistics, p-values and 95% CIs."""
    ci = fit.conf_int()
    return pd.DataFrame(
        {
            "estimate": fit.params,
            "std_error": fit.bse,
            "t": fit.tvalues,
            "p": fit.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )


def permutation_test_term(
    df: pd.DataFrame,
    formula: str,
    term: str,
    weights: np.ndarray | None = None,
    n_perm: int = 5000,
    seed: int = SEED,
) -> dict:
    """Permutation test for one coefficient, holding the rest of the model fixed.

    Residuals of the model without `term` are permuted (Freedman-Lane), so the
    null distribution of the t statistic is built without assuming a normal
    error distribution.  Exchangeability is assessed on the standardised
    residuals, which is why the weights are passed in.
    """
    rng = np.random.default_rng(seed)
    fitter = (lambda d: smf.wls(formula, data=d, weights=weights).fit()) if weights is not None \
        else (lambda d: smf.ols(formula, data=d).fit())
    full = fitter(df)
    param = _match_param(full, term)
    observed = float(full.tvalues[param])

    reduced_formula = _drop_term(formula, term)
    reduced_fitter = (lambda d: smf.wls(reduced_formula, data=d, weights=weights).fit()) if weights is not None \
        else (lambda d: smf.ols(reduced_formula, data=d).fit())
    reduced = reduced_fitter(df)
    fitted, resid = np.asarray(reduced.fittedvalues), np.asarray(reduced.resid)

    null = np.empty(n_perm)
    work = df.copy()
    for i in range(n_perm):
        work["y"] = fitted + rng.permutation(resid)
        null[i] = float(fitter(work).tvalues[param])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {"term": term, "t_observed": observed, "p_permutation": float(p), "n_perm": n_perm}


def _match_param(fit, term: str) -> str:
    """Resolve a formula term to its parameter name (patsy inserts spaces in I(...))."""
    key = term.replace(" ", "")
    for name in fit.params.index:
        if name.replace(" ", "") == key:
            return name
    raise KeyError(f"term {term!r} is not a parameter of the fitted model")


def _drop_term(formula: str, term: str) -> str:
    lhs, rhs = formula.split("~")
    parts = [p.strip() for p in rhs.split("+")]
    key = term.replace(" ", "")
    kept = [p for p in parts if p.replace(" ", "") != key]
    if len(kept) == len(parts):
        raise ValueError(f"term {term!r} not found as a top-level term of {formula!r}")
    return f"{lhs.strip()} ~ " + " + ".join(kept)


def bootstrap_coefficients(
    df: pd.DataFrame,
    formula: str,
    variance_predictor: str = "x4",
    n_boot: int = 2000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Percentile bootstrap CIs that re-estimate the variance model each resample.

    Rows are resampled with replacement (pairs bootstrap), so the intervals do
    not rely on the variance function being exactly right.
    """
    rng = np.random.default_rng(seed)
    draws = []
    n = len(df)
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, n)]
        ols = smf.ols(formula, data=sample).fit()
        w = variance_weights(variance_model(ols, sample, variance_predictor))
        draws.append(smf.wls(formula, data=sample, weights=w).fit().params)
    boot = pd.DataFrame(draws)
    return pd.DataFrame(
        {
            "boot_sd": boot.std(ddof=1),
            "boot_ci_low": boot.quantile(0.025),
            "boot_ci_high": boot.quantile(0.975),
        }
    )
