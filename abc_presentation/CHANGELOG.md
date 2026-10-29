# Changelog

## 2026-10-29 — initial release

Added the ABC / Lotka-Volterra analysis supporting the class presentation.

- `lotka_volterra.py`: vectorised fixed-step RK4 solver for the predator-prey
  system, seeded synthetic data generation, and the expected-residual noise floor.
- `abc_inference.py`: rejection ABC scoring all noise levels off one set of
  simulations; a retained prior sample so tolerances can be re-thresholded
  without re-simulating; the exact grid posterior used as the reference; and
  ABC-SMC with the Beaumont (2009) twice-the-variance perturbation kernel.
- `abc_figures.py`: data and posterior-predictive bands, ABC against the exact
  posterior, and the tolerance versus cost trade-off.
- `abc_analysis.ipynb`: annotated entry point.

Correction made during development, recorded here because it changes the
interpretation: an earlier version added fresh simulation noise to each simulated
trajectory *and* compared against the already-noisy data. The achievable distance
floor in that formulation is `sigma*sqrt(4n)`, not `sigma*sqrt(2n)`, so the
tolerance schedule was driven below the floor, acceptance collapsed to 0.02%, and
the apparently sharp posterior was chasing the noise realisation. The released
code compares the deterministic solution against the noisy data, so the floor is
the fixed and achievable `||e_obs||`.
