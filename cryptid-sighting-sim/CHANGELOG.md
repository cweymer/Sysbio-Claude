# Changelog

## 2026-10-16
- Initial release of the Cascadia Cryptid Sighting Survey teaching dataset.
- `simulate_cryptids.py`: negative binomial simulation, 500 park-months, four
  predictors plus an exposure offset, with ground-truth constants in `TRUTH`
  and `DISTRICT_EFFECT`.
- Exposure relabelled from visitor-hours to hundreds of visitor-hours so the
  implied sighting rate is plausible (0.53 per 100 visitor-hours). Generative
  math unchanged; the offset is still log of that column.
- `README.md`: story, parameter table, identified-contrast analysis, eight
  documented traps with the misfit output each produces, and classroom notes.
- `trap_demos.ipynb`: entry point reproducing the true fit and each wrong fit.
- Notebook ships unexecuted (no kernel in the build sandbox); all cell code was
  run in-process and reproduces the numbers quoted in README.md and fits.txt.
- Fix: README trap 2 quoted an exposure range of 1 to ~400 (the clip target in
  the generating code). The realised column runs 4 to 211, median 28.

## 2026-10-16 (later)
- Added `correct_analysis.ipynb`: the worked correct analysis in seven steps, with the
  dispersion estimated rather than assumed, the collinear vendor pair rotated into
  average and difference, likelihood ratio tests for functional form and for the
  district block, rate ratios with intervals, and randomised quantile residuals.
- Added `analysis_plots.py` and `figures/`: six figures following the project
  visualisation conventions (Okabe-Ito palette, Arial, editable SVG plus PNG, units on
  axis labels, legends top right, error bars on all binned summaries).
- Fixed two figure issues found on first render: per-row colours in the forest plot
  needed drawing individually, and the rate-ratio axis needed plain tick labels in
  place of scientific notation.
- The mean-variance figure now draws the NB curve at the FITTED size, not the
  generating value, so the notebook never uses the ground truth to justify a choice.
