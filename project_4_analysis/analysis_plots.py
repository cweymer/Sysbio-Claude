"""Figures for the project 4 analysis.

Every figure follows the project's visualisation conventions: the Okabe-Ito
colour-blind-safe palette, Arial text, axis labels with units where they exist,
legends in the upper right, and both an editable SVG and a raster PNG written
into `figures/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

FIGDIR = Path("figures")


def set_style() -> None:
    """Publication defaults: Arial text, editable SVG text, light grid."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def save(fig, name: str, figdir: Path | str = FIGDIR) -> Path:
    """Write `name`.svg (editable) and `name`.png (raster) into `figdir`."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    svg = figdir / f"{name}.svg"
    fig.savefig(svg, format="svg")
    fig.savefig(figdir / f"{name}.png", format="png")
    return svg


def _lowess(y, x, frac=0.5):
    out = sm.nonparametric.lowess(np.asarray(y), np.asarray(x), frac=frac)
    return out[:, 0], out[:, 1]


def plot_distributions(df: pd.DataFrame, columns: list[str], figdir=FIGDIR):
    """Histograms of every column, each panel starting at a zero count baseline."""
    fig, axes = plt.subplots(1, len(columns), figsize=(2.1 * len(columns), 2.3), constrained_layout=True)
    for ax, col in zip(np.atleast_1d(axes), columns):
        ax.hist(df[col], bins=30, color=OKABE_ITO["sky"], edgecolor="white", linewidth=0.4)
        ax.set_xlabel(f"{col} (arbitrary units)")
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].set_ylabel("Number of observations")
    fig.suptitle("Marginal distributions of predictors and response", fontsize=10)
    save(fig, "fig1_distributions", figdir)
    return fig


def plot_response_vs_predictors(df: pd.DataFrame, predictors: list[str], figdir=FIGDIR):
    """Response against each predictor with a LOWESS trend on a shared y axis."""
    fig, axes = plt.subplots(1, len(predictors), figsize=(2.1 * len(predictors), 2.4),
                             sharey=True, constrained_layout=True)
    pad = 0.05 * (df["y"].max() - df["y"].min())
    for ax, col in zip(axes, predictors):
        ax.scatter(df[col], df["y"], s=7, alpha=0.45, color=OKABE_ITO["blue"],
                   edgecolors="none", label="Observation")
        gx, gy = _lowess(df["y"], df[col])
        ax.plot(gx, gy, color=OKABE_ITO["vermillion"], linewidth=1.8, label="LOWESS trend")
        ax.set_xlabel(f"{col} (arbitrary units)")
    axes[0].set_ylabel("y (arbitrary units)")
    axes[0].set_ylim(df["y"].min() - pad, df["y"].max() + pad)
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Response against each predictor", fontsize=10)
    save(fig, "fig2_response_vs_predictors", figdir)
    return fig


def plot_additive_diagnostics(fit, df: pd.DataFrame, figdir=FIGDIR):
    """Three-panel diagnostic set for the additive linear fit."""
    resid = np.asarray(fit.resid)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(fit.fittedvalues, resid, s=7, alpha=0.45, color=OKABE_ITO["blue"], edgecolors="none")
    ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax.set_xlabel("Fitted value of y")
    ax.set_ylabel("Residual (y units)")
    ax.set_title("Residuals against fitted values")

    ax = axes[1]
    ax.scatter(df["x3"], resid, s=7, alpha=0.45, color=OKABE_ITO["blue"],
               edgecolors="none", label="Residual")
    gx, gy = _lowess(resid, df["x3"])
    ax.plot(gx, gy, color=OKABE_ITO["vermillion"], linewidth=1.8, label="LOWESS trend")
    ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax.set_xlabel("x3 (arbitrary units)")
    ax.set_ylabel("Residual (y units)")
    ax.set_title("Curvature in x3")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.scatter(df["x4"], np.abs(resid), s=7, alpha=0.45, color=OKABE_ITO["green"],
               edgecolors="none", label="Absolute residual")
    gx, gy = _lowess(np.abs(resid), df["x4"])
    ax.plot(gx, gy, color=OKABE_ITO["vermillion"], linewidth=1.8, label="LOWESS trend")
    ax.set_xlabel("x4 (arbitrary units)")
    ax.set_ylabel("Absolute residual (y units)")
    ax.set_ylim(bottom=0)
    ax.set_title("Spread grows with x4")
    ax.legend(loc="upper right", frameon=False)

    save(fig, "fig3_additive_model_diagnostics", figdir)
    return fig


def plot_interaction(df: pd.DataFrame, figdir=FIGDIR):
    """Response against x1 within terciles of x2, which exposes the product term."""
    tercile = pd.qcut(df["x2"], 3, labels=["low", "middle", "high"])
    colours = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    for level, colour in zip(["low", "middle", "high"], colours):
        sub = df[tercile == level]
        ax.scatter(sub["x1"], sub["y"], s=8, alpha=0.5, color=colour, edgecolors="none")
        slope, intercept = np.polyfit(sub["x1"], sub["y"], 1)
        xs = np.linspace(sub["x1"].min(), sub["x1"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=colour, linewidth=1.8,
                label=f"x2 {level} (slope {slope:.2f})")
    ax.set_xlabel("x1 (arbitrary units)")
    ax.set_ylabel("y (arbitrary units)")
    ax.set_title("Slope of y on x1 reverses with x2")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig4_x1_by_x2_interaction", figdir)
    return fig


def plot_variance_function(df: pd.DataFrame, resid: np.ndarray, n_bins: int = 6, figdir=FIGDIR):
    """Binned residual standard deviation against x4, with bootstrap error bars."""
    bins = pd.qcut(df["x4"], n_bins)
    rng = np.random.default_rng(0)
    centres, sds, los, his = [], [], [], []
    for interval, idx in pd.Series(resid, index=df.index).groupby(bins, observed=True).groups.items():
        r = np.asarray(pd.Series(resid, index=df.index).loc[idx])
        boot = [np.std(rng.choice(r, r.size, replace=True), ddof=1) for _ in range(1000)]
        centres.append(df.loc[idx, "x4"].mean())
        sds.append(r.std(ddof=1))
        los.append(np.quantile(boot, 0.025))
        his.append(np.quantile(boot, 0.975))
    centres, sds = np.array(centres), np.array(sds)
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    ax.errorbar(centres, sds, yerr=[sds - np.array(los), np.array(his) - sds],
                fmt="o", color=OKABE_ITO["purple"], capsize=3, markersize=4,
                label="Binned residual SD (95% bootstrap CI)")
    ax.set_xlabel("x4 (arbitrary units)")
    ax.set_ylabel("Residual standard deviation (y units)")
    ax.set_ylim(bottom=0)
    ax.set_title("Error scale increases with x4")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig5_variance_function", figdir)
    return fig


def plot_final_diagnostics(std_resid: np.ndarray, df: pd.DataFrame, figdir=FIGDIR):
    """Histogram, Q-Q plot and spread check for the standardised residuals."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.hist(std_resid, bins=30, color=OKABE_ITO["sky"], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Standardised residual")
    ax.set_ylabel("Number of observations")
    ax.set_ylim(bottom=0)
    ax.set_title("Distribution after weighting")

    ax = axes[1]
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    ax.scatter(osm, osr, s=7, alpha=0.5, color=OKABE_ITO["blue"], edgecolors="none",
               label="Standardised residual")
    lim = [min(osm.min(), osr.min()) - 0.3, max(osm.max(), osr.max()) + 0.3]
    ax.plot(lim, lim, color=OKABE_ITO["black"], linewidth=0.9, label="Normal reference")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Observed quantile")
    ax.set_title("Normal Q-Q plot")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.scatter(df["x4"], np.abs(std_resid), s=7, alpha=0.45, color=OKABE_ITO["green"],
               edgecolors="none", label="Absolute residual")
    gx, gy = _lowess(np.abs(std_resid), df["x4"])
    ax.plot(gx, gy, color=OKABE_ITO["vermillion"], linewidth=1.8, label="LOWESS trend")
    ax.set_xlabel("x4 (arbitrary units)")
    ax.set_ylabel("Absolute standardised residual")
    ax.set_ylim(bottom=0)
    ax.set_title("Spread is now flat in x4")
    ax.legend(loc="upper right", frameon=False)

    save(fig, "fig6_final_model_diagnostics", figdir)
    return fig


def plot_fitted_effects(fit, df: pd.DataFrame, figdir=FIGDIR):
    """Fitted partial effects of the three retained structural terms."""
    params = fit.params
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)

    ax = axes[0]
    grid = np.linspace(df["x1"].min(), df["x1"].max(), 100)
    for level, colour in zip([-1.0, 0.0, 1.0],
                             [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["vermillion"]]):
        ax.plot(grid, params["x1"] * grid + params["x1:x2"] * grid * level,
                color=colour, linewidth=1.8, label=f"x2 = {level:+.0f}")
    ax.set_xlabel("x1 (arbitrary units)")
    ax.set_ylabel("Partial effect on y (y units)")
    ax.set_title("x1 effect depends on x2")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    grid = np.linspace(df["x3"].min(), df["x3"].max(), 100)
    ax.plot(grid, params["x3"] * grid + params["I(x3 ** 2)"] * grid**2,
            color=OKABE_ITO["green"], linewidth=1.8, label="Fitted quadratic")
    ax.set_xlabel("x3 (arbitrary units)")
    ax.set_ylabel("Partial effect on y (y units)")
    ax.set_title("x3 enters as a symmetric U")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    grid = np.linspace(df["x4"].min(), df["x4"].max(), 100)
    ax.plot(grid, params["x4"] * grid, color=OKABE_ITO["purple"], linewidth=1.8,
            label="Fitted linear term")
    ax.set_xlabel("x4 (arbitrary units)")
    ax.set_ylabel("Partial effect on y (y units)")
    ax.set_title("x4 effect is linear")
    ax.legend(loc="upper right", frameon=False)

    save(fig, "fig7_fitted_effects", figdir)
    return fig
