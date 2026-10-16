# The Cascadia Cryptid Sighting Survey

A teaching dataset with a known generating mechanism, built so that an analyst
who does everything by reflex gets a confident, publishable, wrong answer.

## The story

The Cascadia Interagency Trail Office has spent four years running a reporting
form at trailhead kiosks: *"Did you see, hear, or smell anything you could not
identify?"* Five ranger districts, 500 park-months of returns. Management wants
to know what drives Bigfoot sighting reports, and three things happened on the
way to the dataset.

First, nobody could agree on how to measure forest cover, so the office bought
canopy estimates from **two satellite vendors**. Vendor A and Vendor B score the
same pixels with different proprietary algorithms and disagree politely by a few
percentage points. Both columns are in the file. Neither was dropped, because
the contracts were already signed.

Second, the office's economist insisted on including **campground beer sales**,
and was right to — but not in the direction anyone expected. Sighting reports
are high among the dawn birdwatching crowd (no beer, binoculars, genuinely out
there at 5 a.m.), fall off among moderate drinkers (who are asleep by the time
anything interesting happens), and climb steeply again at the top of the
distribution (for reasons that need no elaboration). The effect is U-shaped.

Third, and fatally, the busiest park-months are also the ones with the most
reports, because **more visitors file more forms**. The exposure column is
right there in the file, labelled in hundreds of visitor-hours, and it is the
single most important variable in the dataset. It is not a predictor. It is a
denominator.

## Files

| file | function |
|---|---|
| `simulate_cryptids.py` | generating code; `python simulate_cryptids.py` rewrites the CSV |
| `cryptid_sightings.csv` | the dataset, 500 rows |
| `trap_demos.ipynb` | entry point: fits the true model and each wrong model side by side |
| `correct_analysis.ipynb` | the worked correct analysis, with figures |
| `analysis_plots.py` | figure code (Okabe-Ito palette, Arial, SVG output) |
| `figures/` | six figures, SVG and PNG |
| `fits.txt` | saved output of the trap fits, for reference |
| `CHANGELOG.md` | change log |

## Columns

| column | meaning |
|---|---|
| `park_month_id` | row key |
| `canopy_vendor_a` | canopy cover, percent, Vendor A |
| `canopy_vendor_b` | canopy cover, percent, Vendor B |
| `beer_sales_z` | campground beer sales, z-scored |
| `ranger_district` | district label, integer 1-5 (**unordered**) |
| `visitor_hundred_hours` | exposure, hundreds of visitor-hours (**offset**) |
| `sightings` | reports filed (outcome, counts) |

## True structure

Negative binomial, log link, size **theta = 1.6** (variance = mu + mu^2/1.6),
with a log-exposure offset whose coefficient is fixed at 1:

```
log(mu) = -1.20
        + 0.330 * (canopy_vendor_a - 50)/10
        - 0.295 * (canopy_vendor_b - 50)/10
        - 0.280 * beer_sales_z  +  0.210 * beer_sales_z^2
        + district effect
        + 1.0 * log(visitor_hundred_hours)        <- offset, not a free slope

district: 1 Mt. Humptulips   0.00 (reference)
          2 Sasquatch Flats  +0.62
          3 Grey Owl Basin   -0.41
          4 Cougar Creek     +0.55
          5 Lake Quinault    -0.15
```

Realised data: mean 20.4 sightings, median 9, max 351, 4.8 percent zeros,
variance/mean = 52.0, overall rate 0.53 sightings per 100 visitor-hours.
Correlations: A/B 0.974, A/beer 0.573, B/log-exposure 0.218.
VIFs: vendor A 19.7, vendor B 19.3, beer 1.5, district 1.0.
The beer U-shape has its minimum at z = +0.67.

## What is and is not identified

Fitting the exactly correct model:

| term | truth | est | SE | p |
|---|---|---|---|---|
| intercept | -1.20 | -1.255 | 0.095 | <0.001 |
| vendor A (scaled) | +0.330 | +0.134 | 0.163 | 0.41 |
| vendor B (scaled) | -0.295 | -0.038 | 0.157 | 0.81 |
| beer | -0.280 | -0.275 | 0.048 | <0.001 |
| beer^2 | +0.210 | +0.230 | 0.031 | <0.001 |
| district 2 | +0.62 | +0.591 | 0.120 | <0.001 |
| district 3 | -0.41 | -0.405 | 0.127 | 0.001 |
| district 4 | +0.55 | +0.736 | 0.124 | <0.001 |
| district 5 | -0.15 | -0.079 | 0.124 | 0.52 |

At r = 0.974 the two vendors carry information about their **common component**
only:

- sum (A + B): truth 0.035, est 0.096, SE 0.045 — well estimated
- difference (A - B): truth 0.625, est 0.172, SE 0.317 — uninformative

So the correct model returns two non-significant, badly-off canopy coefficients.
That is not a failure of the fit. It is the honest answer, and reporting it is
the skill being taught.

## Traps, with the damage each one does

**1. Poisson instead of negative binomial.** The correctly specified Poisson fit
still has Pearson chi2/df = 10.1, and its standard errors are 0.26x the NB ones.
Null predictors print as highly significant. *Tell:* variance/mean = 52.

**2. Treating exposure as a predictor, or omitting it.** Drop
`visitor_hundred_hours` and the vendor block inverts: vendor B becomes
**+0.357, p = 0.019** against a true **-0.295**, while vendor A goes to -0.041,
p = 0.80 against a true +0.330. Both the sign and the significance of the
"winning" predictor are manufactured by exposure confounding. Entering raw
exposure linearly is equally wrong — the realised column runs 4 to 211
(median 28), a 53-fold spread.

**3. `ranger_district` as a continuous number.** The true effects are
non-monotonic, so a single slope averages them away: est -0.015, p = 0.58,
"district doesn't matter". Coded as a factor, the 4-df likelihood-ratio test
gives a deviance drop of 124.0. Nothing in the CSV flags this column as a label.

**4. Assuming beer sales act linearly.** A linear-only term gives -0.414,
p < 1e-10: a confident monotone story about sobriety and credulity that is the
wrong shape. The curvature also leaks into the canopy block, because beer and
vendor A correlate at 0.57.

**5. "Fixing" collinearity by dropping a variable.** VIF ~20 invites it. Drop
vendor B from the otherwise-correct model and vendor A becomes **+0.096,
p = 0.034** — significant, tidy, and equal to neither 0.330 nor the identified
sum of 0.035. Stepwise selection, LASSO, and "keep the lower-VIF one" all land
here silently.

**6. Reporting per-coefficient verdicts at all.** An analyst can get the family,
the offset, the factor coding and the curvature all correct and still write a
wrong conclusion, because the pair is not separately identified. "Vendor A and
vendor B have no effect on sightings" is as wrong as picking one.

**7. Reading the tail as outliers.** Max 351 against a median of 9. Those rows
are high-exposure, high-district months and are fully in-model. Winsorising,
log(y+1) with OLS, or deleting "outliers" removes real signal. The 4.8 percent
zeros are ordinary NB zeros: there is **no zero-inflation component**, so a
significant Vuong test is a false positive driven by the unmodelled curvature
and district structure.

**8. Scale.** Canopy slopes apply to (percent - 50)/10. In raw percentage units
they are ~0.033 and ~-0.030, which is easy to dismiss as negligible next to
beer's -0.28 when the units differ by a factor of ten.

## The worked correct analysis

`correct_analysis.ipynb` is the companion to the trap list: the analysis done properly,
in the order the decisions actually arise, with figures in `figures/` (SVG for editing,
PNG for slides). Plotting code is in `analysis_plots.py`.

| step | decision | what the data say |
|---|---|---|
| 1 | rows are unequal-sized park-months, so estimate a *rate* | exposure 4 to 211, 53 fold spread |
| 2 | choose the family from the mean-variance relationship | variance/mean 52; Poisson chi2/df 10.1; fitted size 1.82 |
| 3 | rotate the collinear pair into average and disagreement | VIF 19.7 and 19.3 become 1.5 and 1.0 |
| 4 | test functional form for beer sales | quadratic LR chi2 64.8 on 1 df, minimum at z = 0.60 |
| 5 | test district as a 4-df block, never as a slope | LR chi2 122.5 (p = 1.5e-25) versus p = 0.91 as a slope |
| 6 | report rate ratios, flag what is unidentified | canopy average 1.10 (1.01 to 1.20); disagreement 1.01 (0.98 to 1.04) |
| 7 | check fit with randomised quantile residuals | mean 0.00, SD 1.00, Shapiro-Wilk p = 0.54 |

The pivotal move is step 3. Instead of deleting a vendor to satisfy a VIF rule, replace
the pair with **their average** and **their difference**. This is a rotation, not a
different model — the log-likelihood is identical to the original parameterisation
(-1760.253 either way) — but it puts the identified quantity and the unidentified one on
separate, orthogonal axes, so the output says which is which instead of hiding it.
The vendor average is a clear effect; the disagreement term is uninformative, exactly as
it should be if the vendors differ only by measurement noise.

Note that the dispersion is **estimated** in this notebook (alpha = 0.55, size 1.82)
rather than set to the generating value of 1.6, as it must be in a real analysis. The
fitted canopy, beer and district estimates in the table above therefore differ slightly
from the known-alpha fits quoted earlier in this README.

## An analysis that passes

1. Check dispersion before choosing a family; fit NB, or quasi-Poisson if only
   the standard errors matter.
2. Put `log(visitor_hundred_hours)` in as an offset, after asking whether the
   quantity of interest is a rate.
3. Code `ranger_district` as a factor and test it with a 4-df LR test.
4. Allow curvature in `beer_sales_z` (quadratic or spline), compare by LR/AIC.
5. Diagnose the vendor block, report the identified sum with its CI, and state
   plainly that the individual slopes are not separable at r = 0.974.
6. Expected conclusion: strong exposure scaling, a real U-shape in beer sales, a
   clear district effect, and no resolvable separate contribution from either
   vendor.

## Using it in class

Hand out the CSV and the column table only. Ask for "the drivers of sighting
reports" with no further guidance, then collect the answers before showing this
README. The common first submission reports beer sales as a strong negative
linear effect, one vendor as significant, and district as irrelevant — three
wrong answers out of three, each from a separate and individually reasonable
shortcut. The point is that none of the eight traps requires a mistake in
arithmetic; every one is a modelling decision made by default.
