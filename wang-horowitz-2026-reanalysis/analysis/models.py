"""Model fits for the Wang & Horowitz (2026) re-analysis.

Each function fits one model at the correct unit of replication and returns a
tidy frame. Nothing here aggregates away between-dog variance: `dog` enters
every model either as a random intercept or as the row index of a dog-level
summary.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def fit_part1_lmm(df, outcome="investigation_s", dog="dog"):
    """Part 1 investigation time: one mixed model instead of pooled Wilcoxons.

    log1p transform handles the right skew and floor at zero while keeping the
    model on a scale where a presentation-by-pairing interaction is meaningful.
    Dog is a random intercept, which is what removes the ~3x pseudoreplication
    of treating 122 trials from 43 dogs as independent.
    """
    d = df.copy()
    d["log_time"] = np.log1p(d[outcome])
    model = smf.mixedlm("log_time ~ C(presentation) * C(pairing) + C(owner_enhanced) + C(period)",
                        data=d, groups=d[dog])
    res = model.fit(reml=True)
    return res


def fit_preference_index(df, dog="dog"):
    """Dual-box phase as a preference, not as two coupled durations.

    A4 and B are measured simultaneously under a near-fixed time budget, so a
    paired test on them is a preference test wearing the wrong clothes. Beta
    regression on B / (A4 + B) with an intercept tested against 0.5 is the
    model that matches the measurement.
    """
    from statsmodels.othermod.betareg import BetaModel
    d = df.dropna(subset=["pref_index"]).copy()
    # Smithson-Verkuilen squeeze keeps the response strictly inside (0, 1)
    n = len(d)
    d["y"] = (d["pref_index"] * (n - 1) + 0.5) / n
    exog = sm.add_constant(pd.DataFrame(index=d.index))
    res = BetaModel(d["y"], exog, exog_precision=exog).fit()
    # Test the mean against 0.5 on the logit scale, i.e. intercept against 0
    est = res.params["const"]
    se = res.bse["const"]
    z = est / se
    from scipy import stats as st
    return {"logit_mean": float(est), "se": float(se), "z": float(z),
            "p": float(2 * st.norm.sf(abs(z))),
            "mean_preference": float(1 / (1 + np.exp(-est))),
            "n_dogs": int(d[dog].nunique())}


def fit_part2_binomial_glmm(df, outcome, dog="dog", condition="condition",
                            extra_terms=""):
    """Part 2 behaviour: binomial GLMM on the sample-point-level binary data.

    This is the analysis the published Part 2 should have run. It keeps the
    12 within-trial sample points as information while putting the dog in the
    error term, instead of averaging across dogs and testing 12 bin means as
    if they were replicates. Variational Bayes is used because 9/9/6 dogs will
    not support reliable Laplace asymptotics; the dog-level permutation test
    is the primary analysis and this is the model-based companion.

    `extra_terms` lets the exploratory sex x condition model be fitted through
    the same code path, e.g. extra_terms=" * C(sex)".
    """
    d = df.dropna(subset=[outcome]).copy()
    d["dog_idx"] = pd.Categorical(d[dog]).codes
    d["_y"] = d[outcome].astype(int)
    fml = f"_y ~ C({condition}){extra_terms}"
    vc = {"dog": "0 + C(dog_idx)"}
    model = BinomialBayesMixedGLM.from_formula(fml, vc_formulas=vc, data=d,
                                               vcp_p=2.0)
    return model.fit_vb(verbose=False)


def glmm_table(res):
    """Tidy fixed-effect estimates from a BinomialBayesMixedGLM VB fit."""
    k = len(res.model.exog_names)
    return pd.DataFrame({
        "term": res.model.exog_names,
        "estimate_logit": res.params[:k],
        "sd": res.cov_params()[:k] ** 0.5 if res.cov_params().ndim == 1
              else np.sqrt(np.diag(res.cov_params()))[:k],
    }).assign(odds_ratio=lambda t: np.exp(t.estimate_logit))


def fit_posture_multinomial(df):
    """Three mutually exclusive postures as one compositional model.

    Testing upright, sitting and prone with three separate omnibus tests
    triple-counts a single compositional shift. One multinomial fit over the
    posture triple replaces them.
    """
    d = df.dropna(subset=["posture"]).copy()
    y = pd.Categorical(d["posture"]).codes
    X = pd.get_dummies(d["condition"], drop_first=True).astype(float)
    X = sm.add_constant(X)
    res = sm.MNLogit(y, X).fit(disp=False)
    return res


def dog_level_table(df, value, condition="condition", dog="dog"):
    """One row per dog: the unit every Part 2 test should have used."""
    g = df.groupby([dog, condition], observed=True)[value]
    return g.mean().reset_index().rename(columns={value: f"{value}_dog_mean"})
