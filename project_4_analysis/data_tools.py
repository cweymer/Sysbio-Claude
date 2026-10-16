"""Loading and exploratory summaries for the project 4 dataset.

The dataset is a 500-row table of four continuous predictors (x1-x4) and one
continuous response (y).  Nothing about the generating process is assumed here;
every function returns a table the analyst inspects before modelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["x1", "x2", "x3", "x4"]
RESPONSE = "y"


def load_dataset(path: str | Path = "project_4_dataset.csv") -> pd.DataFrame:
    """Read the CSV and check that the expected columns are present and finite."""
    df = pd.read_csv(path)
    missing = set(PREDICTORS + [RESPONSE]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if not np.isfinite(df[PREDICTORS + [RESPONSE]].to_numpy()).all():
        raise ValueError("dataset contains non-finite values")
    return df


def summarise_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column location, spread, shape and a Shapiro-Wilk normality p-value.

    The Shapiro p-value is a description of the marginal shape only.  Marginal
    normality of a predictor is not an assumption of linear regression, and
    marginal non-normality of the response is not by itself a problem; both are
    reported so the analyst can see which columns are symmetric and which are not.
    """
    rows = []
    for col in PREDICTORS + [RESPONSE]:
        v = df[col].to_numpy()
        rows.append(
            {
                "variable": col,
                "n": v.size,
                "mean": v.mean(),
                "sd": v.std(ddof=1),
                "min": v.min(),
                "median": np.median(v),
                "max": v.max(),
                "skew": stats.skew(v),
                "excess_kurtosis": stats.kurtosis(v),
                "shapiro_p": stats.shapiro(v).pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation of each predictor with the response.

    Both are reported because a predictor that enters the response through a
    symmetric nonlinearity or through a product with another predictor has a
    marginal correlation near zero under either measure; agreement between the
    two therefore does not license dropping a variable.
    """
    rows = []
    for col in PREDICTORS:
        pear = stats.pearsonr(df[col], df[RESPONSE])
        spear = stats.spearmanr(df[col], df[RESPONSE])
        rows.append(
            {
                "predictor": col,
                "pearson_r": pear.statistic,
                "pearson_p": pear.pvalue,
                "spearman_rho": spear.statistic,
                "spearman_p": spear.pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("predictor")


def predictor_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations among the predictors (first look for collinearity)."""
    return df[PREDICTORS].corr()


def fdr_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * p.size / (np.arange(p.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return pd.Series(out, index=pvalues.index, name="p_fdr")
