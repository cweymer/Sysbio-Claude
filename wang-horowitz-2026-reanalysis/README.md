# Re-analysis plan — Wang & Horowitz (2026)

Working code for the statistical analysis I would run on **Wang, A. & Horowitz, A.
(2026), "Dogs (*Canis familiaris*) distinguish conspecific emotional chemosignals",
*Scientific Reports* 16:11176** (doi:10.1038/s41598-026-41426-1).

This is a **plan expressed as code**, not a replication. The paper's deposited data
(Dryad `doi:10.5061/dryad.bzkh189pt`) is behind an OAuth bearer token and returned
HTTP 401, so the notebook executes against synthetic fixtures built to the structure
the Methods section describes. **Every number it currently produces is a property of
the simulator, not a finding about dogs.** Point `load_deposit` at the real workbooks
and the same cells produce the real result.

## File structure

```
analysis/
  wang_horowitz_reanalysis.ipynb   main notebook: the analysis, in order
  stats_helpers.py                 permutation tests, effect sizes, FDR, ICC/kappa,
                                   sensitivity, and the Kruskal-Wallis H ceiling
  models.py                        mixed model, beta regression, binomial GLMM,
                                   multinomial posture model
  simulate.py                      synthetic fixtures (NOT the paper's data)
  CHANGELOG.md                     change log for the analysis code
tests/
  test_helpers.py                  validates helpers against known values
data/
  README.md                        how to obtain the real deposit
outputs/                           figures and the test register (not committed)
requirements.txt
```

## What this changes relative to the published analysis

| Published | Here |
|---|---|
| Part 2 tested on 12 averaged time bins per condition | Dog-level summaries (9/9/6) as primary, plus a binomial GLMM with `(1\|dog)` on bin-level binary data |
| Pooled Part 1 Wilcoxons over ~122 trials from 43 dogs | One mixed model with dog as a random intercept |
| Paired Wilcoxon on A4 vs B | Beta regression on the preference index B/(A4+B) |
| ~36 p-values, no correction | BH-FDR across a pre-declared confirmatory family; the rest labelled exploratory |
| Wilcoxon V and p only | Hodges-Lehmann shift with bootstrap CI, rank-biserial r, epsilon-squared |
| Normality pre-test selects the test | Family fixed a priori by data type; permutation reference distributions |
| Gaussian GLM on counts out of 12 | Binomial GLMM; one multinomial model for the posture triple |
| No reliability statistic | ICC(2,1) for durations, Cohen's kappa per ethogram item |
| Nulls read as absence | Sensitivity analysis giving the smallest detectable effect |

### The diagnostic that motivated the Part 2 rewrite

`stats_helpers.h_max(sizes)` returns the largest Kruskal-Wallis H attainable for a set
of group sizes under perfect separation. For 9/9/6 dogs it is **20.25**; for three
groups of 12 bins it is **31.14**. The paper reports H = 23.98 and H = 23.11, both
above the dog-level ceiling, which establishes arithmetically that the Part 2 tests
were computed on sample points rather than on animals.

## What no re-analysis can fix

Three problems are in the design. The notebook surfaces them and declines to model
around them:

- **Single odour donor** — n = 1 at the stimulus level, so a species-level claim has no
  error term. Donor cannot enter as a random effect.
- **Emotion confounded with exertion, restraint and collection order** — the joy sample
  is the only one preceded by two minutes of exercise, and every significant
  discrimination contrast involves it.
- **Period, testing room and the owner-distraction task are completely aliased**
  (14+17 = 31 = room 1; 12 = room 2; 17+12 = 29 = the distraction group), so one
  three-level covariate cannot clear any of them individually.

## Running it

```bash
pip install -r requirements.txt
cd analysis && jupyter lab wang_horowitz_reanalysis.ipynb
pytest tests -q          # from the repository root
```

The notebook is committed without stored outputs. Figures and `test_register.csv` are
written to `outputs/`, which is gitignored.

## Provenance

The analysis choices implemented here follow a statistical review of the paper that
established the H-ceiling result, the multiplicity counts (16 of 36 reported p-values
survive Bonferroni, 28 survive BH) and the sensitivity figures (Part 2 has 80% power
only for d >= 1.41-1.60). Those were derived from the manuscript, since the deposit
could not be retrieved.
