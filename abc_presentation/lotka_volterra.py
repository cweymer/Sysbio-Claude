"""Lotka-Volterra forward model and synthetic data generation.

Model (Toni et al. 2009 parameterisation):
    dx/dt = a*x - x*y     prey
    dy/dt = b*x*y - y     predator
"""
import numpy as np


def lv_rk4(a, b, t_obs, x0=1.0, y0=0.5, T=15.0, dt=0.01):
    """Solve the system by fixed-step RK4, vectorised over parameter vectors.

    a, b are array-likes of equal length N. Returns (N, len(t_obs), 2) with
    the prey column first. A fixed step is used rather than an adaptive
    solver so that every parameter draw costs the same and the whole batch
    can be advanced in lockstep as numpy arrays.
    """
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    x = np.full(a.size, x0)
    y = np.full(a.size, y0)
    obs_idx = np.round(np.asarray(t_obs) / dt).astype(int)
    out = np.empty((a.size, len(t_obs), 2))
    ptr = 0
    deriv = lambda x, y: (a * x - x * y, b * x * y - y)
    for step in range(int(round(T / dt)) + 1):
        while ptr < len(obs_idx) and obs_idx[ptr] == step:
            out[:, ptr, 0] = x
            out[:, ptr, 1] = y
            ptr += 1
        k1x, k1y = deriv(x, y)
        k2x, k2y = deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        k3x, k3y = deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
        k4x, k4y = deriv(x + dt * k3x, y + dt * k3y)
        x = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        y = y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
    return out


def make_data(sigmas, t_obs, a_true=1.0, b_true=1.0, seed=7):
    """Noise-free trajectory plus one noisy dataset per sigma.

    Noise is i.i.d. Gaussian on both species. Datasets are drawn in the order
    sigmas are given, from a single seeded generator, so the result is
    reproducible only for a fixed ordering of `sigmas`.
    """
    rng = np.random.default_rng(seed)
    truth = lv_rk4(a_true, b_true, t_obs)[0]
    data = {s: truth + rng.normal(0, s, truth.shape) for s in sigmas}
    return truth, data


def noise_floor(sigma, n_obs):
    """Expected residual norm when the parameters are exactly right.

    This is the smallest distance a deterministic simulation can achieve
    against noisy data, so it is the floor below which the tolerance must
    not be pushed.
    """
    return sigma * np.sqrt(n_obs)
