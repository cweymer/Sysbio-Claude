"""Figures for the correct analysis of the Cascadia Cryptid Sighting Survey.

Okabe-Ito colour-blind safe palette, Arial, SVG output for Illustrator editing.
Each function takes a prepared dataframe (and sometimes a fitted model) and
returns the matplotlib figure, saving an SVG alongside.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}


def set_style():
    """Publication defaults: Arial, editable SVG text, restrained spines."""
    mpl.rcParams.update({
        "font.family": "Arial",
        "svg.fonttype": "none",        # keep text as text for Illustrator
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def _save(fig, name):
    """Write SVG next to the notebook and report the path."""
    fig.savefig(f"figures/{name}.svg")
    fig.savefig(f"figures/{name}.png", dpi=200)
    return fig


def plot_outcome(d, name="fig1_outcome"):
    """Counts and exposure-adjusted rates: look before assuming a family."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(d.sightings, bins=np.arange(0, 360, 10), color=OI["blue"])
    axes[0].set(xlabel="Sighting reports per park month (count)",
                ylabel="Park months (n)", ylim=(0, None))
    axes[0].set_title("A  Raw counts are heavily right skewed", loc="left")
    axes[1].hist(d.visitor_hundred_hours, bins=np.arange(0, 220, 10), color=OI["orange"])
    axes[1].set(xlabel="Exposure (hundreds of visitor hours)",
                ylabel="Park months (n)", ylim=(0, None))
    axes[1].set_title("B  Exposure spans a 53 fold range", loc="left")
    axes[2].hist(d.rate, bins=np.arange(0, 2.1, 0.08), color=OI["green"])
    axes[2].set(xlabel="Reports per 100 visitor hours",
                ylabel="Park months (n)", ylim=(0, None))
    axes[2].set_title("C  The rate is the quantity of interest", loc="left")
    fig.tight_layout()
    return _save(fig, name)


def plot_mean_variance(d, theta, name="fig2_mean_variance", n_bins=12):
    """Binned mean vs variance against the Poisson and NB2 expectations.

    theta is the ESTIMATED size parameter, not the generating value.
    """
    q = pd.qcut(d.fitted_mu, n_bins, duplicates="drop")
    g = d.groupby(q, observed=True).sightings.agg(["mean", "var", "size"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(g["mean"], g["var"], s=28, color=OI["black"],
               label="Observed (binned by fitted mean)", zorder=3)
    grid = np.linspace(g["mean"].min(), g["mean"].max(), 100)
    ax.plot(grid, grid, color=OI["sky"], lw=2, label="Poisson: variance = mean")
    ax.plot(grid, grid + grid**2 / theta, color=OI["vermillion"], lw=2,
            label=f"Negative binomial, fitted size {theta:.2f}")
    ax.set(xscale="log", yscale="log",
           xlabel="Mean reports per park month",
           ylabel="Variance of reports per park month")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, name)


def plot_collinearity(d, name="fig3_collinearity"):
    """Vendor agreement, and why averaging beats dropping one."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].scatter(d.canopy_vendor_a, d.canopy_vendor_b, s=12,
                    color=OI["blue"], alpha=0.55, edgecolors="none")
    lims = (10, 86)
    axes[0].plot(lims, lims, color=OI["black"], lw=1, ls="--", label="Perfect agreement")
    r = d.canopy_vendor_a.corr(d.canopy_vendor_b)
    axes[0].set(xlim=lims, ylim=lims, xlabel="Vendor A canopy cover (%)",
                ylabel="Vendor B canopy cover (%)")
    axes[0].set_title(f"A  Two readings of one quantity (r = {r:.3f})", loc="left")
    axes[0].legend(loc="upper left")
    diff = d.canopy_vendor_a - d.canopy_vendor_b
    axes[1].hist(diff, bins=np.arange(-9, 9.5, 1), color=OI["purple"])
    axes[1].axvline(0, color=OI["black"], lw=1, ls="--")
    axes[1].set(xlabel="Vendor A minus vendor B (percentage points)",
                ylabel="Park months (n)", ylim=(0, None))
    axes[1].set_title(f"B  Disagreement is noise (SD = {diff.std():.2f} pp)", loc="left")
    fig.tight_layout()
    return _save(fig, name)


def plot_beer_shape(d, fit, name="fig4_beer_shape", n_bins=10):
    """Empirical log rate by beer-sales bin, against the fitted quadratic."""
    q = pd.qcut(d.beer_sales_z, n_bins)
    g = d.groupby(q, observed=True).apply(
        lambda s: pd.Series({
            "z": s.beer_sales_z.mean(),
            "log_rate": np.log(s.sightings.sum() / s.visitor_hundred_hours.sum()),
            "se": 1 / np.sqrt(max(s.sightings.sum(), 1)),
        }), include_groups=False)
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.errorbar(g.z, g.log_rate, yerr=1.96 * g.se, fmt="o", ms=5, lw=1,
                capsize=3, color=OI["black"], label="Observed bins (95% CI)", zorder=3)
    grid = np.linspace(d.beer_sales_z.min(), d.beer_sales_z.max(), 200)
    b = fit.params
    curve = b["beer_sales_z"] * grid + b["I(beer_sales_z ** 2)"] * grid**2
    ref = np.interp(0, grid, curve)
    ax.plot(grid, curve - ref + np.interp(0, g.z, g.log_rate),
            color=OI["vermillion"], lw=2, label="Fitted quadratic")
    ax.axvline(-b["beer_sales_z"] / (2 * b["I(beer_sales_z ** 2)"]),
               color=OI["sky"], lw=1.5, ls=":", label="Fitted minimum")
    ax.set(xlabel="Campground beer sales (z score)",
           ylabel="Log reports per 100 visitor hours")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, name)


def plot_coefficients(table, name="fig5_coefficients"):
    """Forest plot of rate ratios with 95% CIs, ordered as given."""
    t = table.iloc[::-1]
    y = np.arange(len(t))
    colors = [OI["vermillion"] if flag else OI["blue"] for flag in t.unidentified]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(t) + 1.6))
    for yi, (_, row), col in zip(y, t.iterrows(), colors):
        ax.plot([row.lo, row.hi], [yi, yi], color=col, lw=1.8,
                solid_capstyle="butt")
    ax.scatter(t.rr, y, s=34, color=colors, zorder=3)
    ax.axvline(1, color=OI["black"], lw=1, ls="--")
    ticks = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    ax.set(yticks=y, yticklabels=t.index, xscale="log",
           xlabel="Rate ratio (95% CI), reports per 100 visitor hours")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.scatter([], [], s=34, color=OI["blue"], label="Identified")
    ax.scatter([], [], s=34, color=OI["vermillion"], label="Not identified (r = 0.974)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, name)


def plot_diagnostics(d, name="fig6_diagnostics", seed=1):
    """Randomised quantile residuals: QQ plot and residuals against fitted mean."""
    from scipy import stats
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    theo = stats.norm.ppf((np.arange(1, len(d) + 1) - 0.5) / len(d))
    axes[0].scatter(theo, np.sort(d.rqr), s=12, color=OI["blue"],
                    alpha=0.6, edgecolors="none")
    lim = (-3.6, 3.6)
    axes[0].plot(lim, lim, color=OI["black"], lw=1, ls="--", label="Identity")
    axes[0].set(xlim=lim, ylim=lim, xlabel="Theoretical normal quantile",
                ylabel="Randomised quantile residual")
    axes[0].set_title("A  Residuals track the reference line", loc="left")
    axes[0].legend(loc="upper left")
    axes[1].scatter(d.fitted_mu, d.rqr, s=12, color=OI["green"],
                    alpha=0.6, edgecolors="none")
    axes[1].axhline(0, color=OI["black"], lw=1, ls="--")
    axes[1].set(xscale="log", xlabel="Fitted mean reports per park month",
                ylabel="Randomised quantile residual", ylim=(-3.6, 3.6))
    axes[1].set_title("B  No trend in mean or spread", loc="left")
    fig.tight_layout()
    return _save(fig, name)
