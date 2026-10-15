"""Reusable statistical helpers for the Wang & Horowitz (2026) re-analysis.

All functions are design-agnostic: they take tidy frames or arrays and return
plain dicts / frames, so the same code serves the real deposit and the
synthetic fixtures used to test the pipeline.

Inference style follows the project convention: permutation tests as the
default where exchangeability is defensible, rank-based effect sizes with
bootstrap intervals, and Benjamini-Hochberg control across a pre-declared
confirmatory family.
"""

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260101)


# ---------------------------------------------------------------- aggregation

def to_dog_level(df, value, by, dog="dog", how="mean"):
    """Collapse within-dog repeated measures to one value per dog per cell.

    This is the step the published analysis omits. `by` is the list of design
    factors that define a cell (e.g. ["condition"] or ["pairing", "presentation"]).
    """
    out = df.groupby([dog] + list(by), observed=True)[value].agg(how).reset_index()
    return out.rename(columns={value: f"{value}_{how}"})


def paired_wide(df, value, within, dog="dog"):
    """Pivot one within-subject factor to wide form, keeping complete cases only.

    Returns the wide frame and the number of dogs dropped, so attrition is
    always reported rather than silently absorbed.
    """
    wide = df.pivot_table(index=dog, columns=within, values=value, observed=True)
    n_before = len(wide)
    wide = wide.dropna()
    return wide, n_before - len(wide)


# ------------------------------------------------------------ permutation tests

def perm_paired(x, y, n_perm=20000, rng=RNG):
    """Exact-in-spirit sign-flip permutation test on paired differences.

    Exchangeability here is the null's own symmetry assumption: under H0 the
    sign of each dog's difference is arbitrary. Reports the observed mean and
    median difference alongside p, since the test alone is not an effect size.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    n = len(d)
    obs = np.mean(d)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"n": n, "mean_diff": obs, "median_diff": float(np.median(d)),
            "p_perm": float(p), "n_perm": n_perm}


def perm_kruskal(groups, n_perm=20000, rng=RNG):
    """Label-permutation Kruskal-Wallis for small, unbalanced groups.

    The asymptotic chi-square reference is unreliable at 9/9/6, which is the
    Part 2 design; permuting group labels gives an exact-conditional p.
    """
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    obs = stats.kruskal(*groups).statistic
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        split, k = [], 0
        for s in sizes:
            split.append(perm[k:k + s]); k += s
        null[i] = stats.kruskal(*split).statistic
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return {"H": float(obs), "p_perm": float(p), "group_sizes": sizes,
            "H_max_possible": float(h_max(sizes)), "n_perm": n_perm}


def h_max(sizes):
    """Maximum attainable Kruskal-Wallis H for the given group sizes.

    Diagnostic for the published Part 2 tests: a reported H above this ceiling
    proves the test was run on a unit other than the one claimed.
    """
    N = sum(sizes)
    ranks, start = [], 1
    for s in sizes:
        ranks.append(np.arange(start, start + s)); start += s
    num = sum(len(r) * (r.mean() - (N + 1) / 2) ** 2 for r in ranks)
    return 12.0 / (N * (N + 1)) * num


# ----------------------------------------------------------------- effect sizes

def hodges_lehmann(x, y=None, alpha=0.05, n_boot=10000, rng=RNG):
    """Hodges-Lehmann location shift with a percentile bootstrap interval.

    This is the estimate that matches a Wilcoxon test; the published paper
    reports V and p with no location estimate at all.
    """
    if y is None:
        d = np.asarray(x, float)
    else:
        d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    walsh = (d[:, None] + d[None, :])[np.triu_indices(len(d))] / 2
    est = float(np.median(walsh))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(d, size=len(d), replace=True)
        w = (s[:, None] + s[None, :])[np.triu_indices(len(s))] / 2
        boot[i] = np.median(w)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"hodges_lehmann": est, "ci_low": float(lo), "ci_high": float(hi)}


def rank_biserial_paired(x, y):
    """Matched-pairs rank-biserial correlation (favourable-rank proportion)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[(~np.isnan(d)) & (d != 0)]
    r = stats.rankdata(np.abs(d))
    pos, neg = r[d > 0].sum(), r[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def epsilon_squared(groups):
    """Epsilon-squared for Kruskal-Wallis: (H - k + 1) / (N - k)."""
    groups = [np.asarray(g, float) for g in groups]
    H = stats.kruskal(*groups).statistic
    N, k = sum(len(g) for g in groups), len(groups)
    return float((H - k + 1) / (N - k))


# ------------------------------------------------------------------ multiplicity

def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg adjusted p-values and rejection flags.

    BH rather than Bonferroni: the contrasts share subjects and habituation
    data, so they are positively dependent, and the goal is discovery.
    """
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return pd.DataFrame({"p_raw": p, "p_bh": out, "reject": out <= q})


# ---------------------------------------------------------- coder reliability

def icc21(ratings):
    """ICC(2,1), two-way random effects, absolute agreement, single rater.

    `ratings` is an n_targets x n_raters array. This is the statistic the paper
    needs for its duration measure and does not report; percentage agreement
    and Pearson r are both inadequate substitutes.
    """
    Y = np.asarray(ratings, float)
    n, k = Y.shape
    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom)


def cohen_kappa(a, b):
    """Cohen's kappa for two coders on a categorical/binary ethogram item."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    tab = np.array([[np.sum((a == i) & (b == j)) for j in cats] for i in cats], float)
    n = tab.sum()
    po = np.trace(tab) / n
    pe = (tab.sum(0) * tab.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe))


# -------------------------------------------------------- sensitivity / power

def min_detectable_d(n1, n2=None, alpha=0.05, power=0.80):
    """Smallest standardised effect detectable at the given power.

    Paired (n2=None) returns d_z against df = n1 - 1; two-sample returns
    Cohen's d against df = n1 + n2 - 2. Solved from the noncentral t rather
    than the normal approximation, which is optimistic at these sample sizes.
    Reported as a sensitivity statement, which is what an absent a priori
    power analysis can still be replaced with at review time.
    """
    from scipy.optimize import brentq
    if n2 is None:
        df, scale = n1 - 1, np.sqrt(n1)
    else:
        df, scale = n1 + n2 - 2, 1.0 / np.sqrt(1 / n1 + 1 / n2)
    crit = stats.t.ppf(1 - alpha / 2, df)

    def shortfall(d):
        # upper tail only: the wrong-sign rejection probability is negligible
        # here and its noncentral-t evaluation is numerically unstable
        achieved = stats.nct.sf(crit, df, d * scale)
        return achieved - power

    return float(brentq(shortfall, 1e-4, 5.0))
