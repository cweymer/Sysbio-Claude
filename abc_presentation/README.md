# Approximate Bayesian Computation on a Lotka-Volterra model

Supporting analysis for the ABC class presentation. Fits the two parameters of a
predator-prey ODE from noisy observations using likelihood-free inference, at
three levels of observation noise, and checks the result against the exact
posterior.

## File structure

| File | Function |
|---|---|
| `abc_analysis.ipynb` | Entry point. Runs the whole analysis in order with the method explained alongside each step. |
| `lotka_volterra.py` | Forward model. Vectorised fixed-step RK4 solver, synthetic data generation, and the noise-floor expression. |
| `abc_inference.py` | Inference. Rejection ABC, a retained prior sample for tolerance sweeps, the exact grid posterior, and ABC-SMC. |
| `abc_figures.py` | The three presentation figures. |
| `CHANGELOG.md` | Change log for this directory. |

Running the notebook writes `fig_lv_data.png`, `fig_lv_posterior.png` and
`fig_lv_tolerance.png` into this directory.

## The model

    dx/dt = a*x - x*y      prey
    dy/dt = b*x*y - y      predator

Truth is `a = b = 1`, `x0 = 1`, `y0 = 0.5`, observed at `t = 1..15` on both
species, giving 30 data points per dataset. Observation noise is i.i.d. Gaussian
at `sigma` = 0.1, 0.5 and 1.0.

## Method notes

- **No summary statistics.** The distance is Euclidean over all 30 observations.
  With a series this short there is nothing to gain by compressing it, and
  compressing would discard phase information.
- **Tolerance.** `eps = sigma * sqrt(2n)`, the residual norm expected from the
  noise model alone. Fixing it this way means it is chosen before any output is
  seen rather than tuned until the answer looks right. It cannot be pushed below
  the realised floor `||e_obs||`.
- **Validation.** The ODE is deterministic, so the Gaussian likelihood is
  available and the exact posterior is evaluated on a 260 x 260 grid. This is the
  reason Lotka-Volterra was chosen: ABC can be checked rather than trusted.

## Reproducing

Needs `numpy`, `scipy`, `pandas`, `matplotlib`, `nbformat`. Run the notebook
top to bottom. The rejection ABC cell draws 6,000,000 parameter pairs and takes
roughly two minutes; everything else is seconds. All generators are explicitly
seeded, so the reported counts and posterior summaries are reproducible.

## Caveats

- `sigma` is treated as known when setting the tolerance; inferring it jointly
  would widen the posteriors.
- The tolerance kernel is a hard indicator rather than a smooth kernel, which is
  why the ABC posteriors sit about 1.5x wider than exact.
- Plain Lotka-Volterra has no prey carrying capacity, so the cycle is neutrally
  stable and its amplitude depends entirely on the initial conditions.
