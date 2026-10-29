"""Rejection ABC, ABC-SMC and the exact reference posterior."""
import numpy as np
from scipy.stats import norm

from lotka_volterra import lv_rk4


def distances(sims, y_obs):
    """Euclidean distance between each simulated trajectory and the data.

    No summary statistics: with 15 time points on 2 species the raw series is
    only 30 numbers, so compressing it would discard phase information for no
    reduction in matching dimension worth having.
    """
    return np.sqrt(np.sum((sims - y_obs) ** 2, axis=(1, 2)))


def rejection_abc(data, t_obs, eps, n_sim, seed=2026, chunk=50_000,
                  prior=(0.0, 5.0)):
    """Rejection ABC against several datasets at once.

    Draws `n_sim` parameter pairs from a uniform prior, solves the ODE once
    per draw, and keeps the draws within `eps[sigma]` of each dataset. Running
    every sigma off one set of simulations is what makes the 6e6-draw budget
    affordable. Returns {sigma: (a_kept, b_kept)} and the total drawn.
    """
    rng = np.random.default_rng(seed)
    lo, hi = prior
    kept = {s: [[], []] for s in data}
    total = 0
    while total < n_sim:
        pa = rng.uniform(lo, hi, chunk)
        pb = rng.uniform(lo, hi, chunk)
        sims = lv_rk4(pa, pb, t_obs)
        for s, y_obs in data.items():
            m = distances(sims, y_obs) <= eps[s]
            if m.any():
                kept[s][0].append(pa[m])
                kept[s][1].append(pb[m])
        total += chunk
    out = {s: (np.concatenate(v[0]), np.concatenate(v[1])) for s, v in kept.items()}
    return out, total


def prior_sample_distances(data, t_obs, n_sim=300_000, seed=101,
                           prior=(0.0, 5.0), chunk=50_000):
    """Keep the parameters and distances of a prior sample.

    Retaining distances rather than only the accepted draws lets the tolerance
    be re-thresholded afterwards, which is how the epsilon sweep is produced
    without re-simulating.
    """
    rng = np.random.default_rng(seed)
    lo, hi = prior
    a_s = rng.uniform(lo, hi, n_sim)
    b_s = rng.uniform(lo, hi, n_sim)
    sims = np.empty((n_sim, len(t_obs), 2))
    for i in range(0, n_sim, chunk):
        j = min(i + chunk, n_sim)
        sims[i:j] = lv_rk4(a_s[i:j], b_s[i:j], t_obs)
    return a_s, b_s, {s: distances(sims, y) for s, y in data.items()}


def exact_posterior_grid(y_obs, sigma, t_obs, a_range=(0.70, 1.35),
                         b_range=(0.40, 1.80), n=260):
    """Exact Gaussian-likelihood posterior evaluated on a parameter grid.

    Available only because the ODE is deterministic, which is the whole reason
    Lotka-Volterra is a good teaching example: ABC can be checked against it.
    Returns (A, B, P) with P normalised to sum to 1.
    """
    ga = np.linspace(*a_range, n)
    gb = np.linspace(*b_range, n)
    A, B = np.meshgrid(ga, gb, indexing="ij")
    sims = np.empty((A.size, len(t_obs), 2))
    flat_a, flat_b = A.ravel(), B.ravel()
    for i in range(0, A.size, 20_000):
        j = min(i + 20_000, A.size)
        sims[i:j] = lv_rk4(flat_a[i:j], flat_b[i:j], t_obs)
    ll = -np.sum((sims - y_obs) ** 2, axis=(1, 2)) / (2 * sigma ** 2)
    P = np.exp(ll - ll.max()).reshape(A.shape)
    return A, B, P / P.sum()


def grid_moments(A, B, P):
    """Posterior mean and sd of each parameter from a normalised grid."""
    ea, eb = float((P * A).sum()), float((P * B).sum())
    sa = float(np.sqrt((P * (A - ea) ** 2).sum()))
    sb = float(np.sqrt((P * (B - eb) ** 2).sum()))
    return ea, sa, eb, sb


def abc_smc(y_obs, t_obs, eps_schedule, n_particles=1000, seed=99,
            prior=(0.0, 5.0), batch=1000):
    """ABC-SMC with a decreasing tolerance schedule (Toni et al. 2009).

    Each population resamples the previous one by weight, perturbs with a
    Gaussian kernel whose variance is twice the population variance (Beaumont
    2009), and reweights by prior / sum_j w_j K(theta | theta_j) to undo the
    fact that proposals no longer come from the prior. Returns the final
    particles, their weights, and a per-population diagnostic table.
    """
    rng = np.random.default_rng(seed)
    lo, hi = prior
    particles = weights = kernel = None
    report = []
    for stage, eps in enumerate(eps_schedule):
        acc_a, acc_b, n_sim, n_acc = [], [], 0, 0
        while len(acc_a) < n_particles:
            if stage == 0:
                pa = rng.uniform(lo, hi, batch)
                pb = rng.uniform(lo, hi, batch)
            else:
                j = rng.choice(n_particles, size=batch, p=weights)
                pa = particles[0][j] + rng.normal(0, kernel[0], batch)
                pb = particles[1][j] + rng.normal(0, kernel[1], batch)
                ok = (pa > lo) & (pa < hi) & (pb > lo) & (pb < hi)
                pa, pb = pa[ok], pb[ok]
            n_sim += pa.size
            m = distances(lv_rk4(pa, pb, t_obs), y_obs) <= eps
            n_acc += int(m.sum())
            acc_a.extend(pa[m].tolist())
            acc_b.extend(pb[m].tolist())
        new = (np.array(acc_a[:n_particles]), np.array(acc_b[:n_particles]))
        if stage == 0:
            w = np.ones(n_particles) / n_particles
        else:
            ka = norm.pdf(new[0][:, None], particles[0][None, :], kernel[0])
            kb = norm.pdf(new[1][:, None], particles[1][None, :], kernel[1])
            w = (1.0 / (hi - lo) ** 2) / ((weights[None, :] * ka * kb).sum(axis=1))
            w /= w.sum()
        particles, weights = new, w
        kernel = (np.sqrt(2 * np.var(new[0])), np.sqrt(2 * np.var(new[1])))
        report.append({"population": stage + 1, "eps": eps, "simulations": n_sim,
                       "acceptance": n_acc / n_sim,
                       "mean_a": float(np.average(new[0], weights=w)),
                       "mean_b": float(np.average(new[1], weights=w)),
                       "ess": float(1 / np.sum(w ** 2))})
    return particles, weights, report
