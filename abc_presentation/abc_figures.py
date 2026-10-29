"""The three figures used in the ABC presentation.

Colour convention is shared across all panels: prey blue, predator red,
ABC particles teal, exact posterior purple, metadata grey.
"""
import numpy as np
import matplotlib.pyplot as plt

from lotka_volterra import lv_rk4

C_PREY, C_PRED = "#1f6feb", "#d1495b"
C_ABC, C_EXACT, C_META = "#2a9d8f", "#7b2cbf", "#6b6b6b"


def _predictive_band(a, b, t_dense, n_show=400, seed=3):
    """95% posterior predictive envelope from a subsample of particles."""
    rng = np.random.default_rng(seed)
    j = rng.choice(a.size, size=min(n_show, a.size), replace=False)
    return np.percentile(lv_rk4(a[j], b[j], t_dense), [2.5, 97.5], axis=0)


def fig_data(truth, data, abc, sigmas, t_obs, eps, path="fig_lv_data.png"):
    """Data, true trajectory and posterior predictive band at each noise level."""
    t_dense = np.linspace(0.01, 15, 400)
    dense = lv_rk4(1.0, 1.0, t_dense)[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, s in zip(axes, sigmas):
        band = _predictive_band(abc[s][0], abc[s][1], t_dense)
        for k, col in ((0, C_PREY), (1, C_PRED)):
            ax.fill_between(t_dense, band[0, :, k], band[1, :, k], color=col, alpha=.25, lw=0)
            ax.plot(t_dense, dense[:, k], color=col, lw=1.2)
            ax.plot(t_obs, data[s][:, k], "o", ms=3.4, color=col, mec="white", mew=.5)
        ax.set_title(f"$\\sigma$ = {s}   ($\\varepsilon$ = {eps[s]:.2f}, {abc[s][0].size} particles)",
                     fontsize=7.5)
        ax.set_xlabel("time")
        ax.set_xticks([0, 5, 10, 15])
        ax.margins(0.04)
    axes[0].set_ylabel("abundance")
    axes[0].text(0.6, 2.6, "prey", color=C_PREY, fontsize=7)
    axes[0].text(4.2, 2.6, "predator", color=C_PRED, fontsize=7)
    axes[0].text(0.6, -1.45, "lines = true trajectory \u00b7 points = data\n"
                             "band = 95% posterior predictive", fontsize=6, color=C_META)
    fig.suptitle("ABC recovers the predator-prey cycle at every noise level; "
                 "the fitted band widens with $\\sigma$", fontsize=8, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def fig_posterior(abc, exact, sigmas, n_show=885, seed=3,
                  path="fig_lv_posterior.png"):
    """ABC particles against exact posterior contours, one panel per sigma."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, s in zip(axes, sigmas):
        a, b = abc[s]
        A, B, P, moments = exact[s]
        j = rng.choice(a.size, size=min(n_show, a.size), replace=False)
        ax.scatter(a[j], b[j], s=4, color=C_ABC, alpha=.40, lw=0)
        ax.contour(A, B, P / P.max(), levels=[0.05, 0.3, 0.7], colors=C_EXACT, linewidths=1.0)
        ax.plot(1.0, 1.0, "+", color="#111111", ms=9, mew=1.6)
        ax.set_title(f"$\\sigma$ = {s}   (sd$_a$: ABC {a.std():.3f} / exact {moments[1]:.3f})",
                     fontsize=7)
        ax.set_xlabel("$a$  (prey growth rate)")
        ax.set_xlim(0.70, 1.35)
        ax.set_ylim(0.40, 1.80)
    axes[0].set_ylabel("$b$  (predator conversion)")
    for dy, lab, col in ((1.70, "ABC particles", C_ABC), (1.60, "exact posterior", C_EXACT),
                         (1.50, "+ true value", "#111111")):
        axes[0].text(0.725, dy, lab, fontsize=6, color=col)
    axes[2].text(0.725, 0.47, f"{n_show} particles shown per panel", fontsize=6, color=C_META)
    fig.suptitle("ABC recovers the exact posterior's centre and the $a$-$b$ trade-off, "
                 "with ~1.5x the spread", fontsize=8, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def fig_tolerance(a_s, b_s, dist, sigma, floor, eps_rule, exact_sd,
                  path="fig_lv_tolerance.png"):
    """Posterior spread and acceptance rate as functions of the tolerance.

    Both failure modes appear here: the sd rises to the prior sd as eps grows,
    and falls below the exact posterior sd as eps approaches the noise floor.
    """
    eps_grid = np.linspace(floor * 1.01, 14, 70)
    sd_a, sd_b, acc = [], [], []
    for e in eps_grid:
        m = dist <= e
        acc.append(max(m.mean(), 1 / dist.size))
        sd_a.append(a_s[m].std() if m.sum() > 5 else np.nan)
        sd_b.append(b_s[m].std() if m.sum() > 5 else np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax1.semilogy(eps_grid, sd_a, color=C_PREY, lw=1.5)
    ax1.semilogy(eps_grid, sd_b, color=C_PRED, lw=1.5)
    ax1.axhline(exact_sd[0], color=C_PREY, lw=.9, ls=(0, (4, 3)))
    ax1.axhline(exact_sd[1], color=C_PRED, lw=.9, ls=(0, (4, 3)))
    ax1.set_ylim(0.008, 3.0)
    ax1.annotate(f"floor $\\|e_{{obs}}\\|$ = {floor:.2f}", xy=(floor, 2.4), xytext=(4.3, 2.4),
                 fontsize=6, color=C_META, va="center",
                 arrowprops=dict(arrowstyle="-", lw=.6, color=C_META))
    ax1.annotate(f"$\\varepsilon=\\sigma\\sqrt{{2n}}$ = {eps_rule:.2f}", xy=(eps_rule, 1.25),
                 xytext=(4.3, 1.25), fontsize=6, color=C_EXACT, va="center",
                 arrowprops=dict(arrowstyle="-", lw=.6, color=C_EXACT))
    ax1.text(12.3, 1.05, "$b$", color=C_PRED, fontsize=8)
    ax1.text(12.3, 0.47, "$a$", color=C_PREY, fontsize=8)
    ax1.text(8.4, 0.030, "dashed = exact posterior sd", fontsize=6, color=C_META, ha="right")
    ax1.set_xlabel("tolerance  $\\varepsilon$")
    ax1.set_ylabel("ABC posterior sd")
    ax1.set_title("Loosening $\\varepsilon$ inflates the posterior", fontsize=7.5)
    ax2.semilogy(eps_grid, acc, color=C_ABC, lw=1.6)
    ax2.set_ylim(5e-4, 2.0)
    ax2.set_xlabel("tolerance  $\\varepsilon$")
    ax2.set_ylabel("acceptance rate")
    ax2.set_title("Tightening $\\varepsilon$ costs simulations", fontsize=7.5)
    ax2.text(13.8, 1.1e-3, f"{100 * (dist <= eps_rule).mean():.2f}% accepted\n"
                           f"at $\\varepsilon=\\sigma\\sqrt{{2n}}$",
             fontsize=6, color=C_META, ha="right")
    for ax in (ax1, ax2):
        ax.axvline(floor, color=C_META, lw=.9)
        ax.axvline(eps_rule, color=C_EXACT, lw=.9, ls=(0, (2, 2)))
    fig.suptitle("Choosing $\\varepsilon$ is the whole game: statistical accuracy against "
                 f"computational cost  ($\\sigma$ = {sigma})", fontsize=8, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
