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
