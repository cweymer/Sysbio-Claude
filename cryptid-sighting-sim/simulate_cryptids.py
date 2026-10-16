"""Simulate the Cascadia Cryptid Sighting Survey.

A teaching dataset: 500 park-months of Bigfoot sighting reports with a known
generating mechanism that is easy to analyse badly. Every trap is deliberate
and documented in README.md. Run as a script to write the CSV.
"""

import numpy as np
import pandas as pd

# --- Ground truth -----------------------------------------------------------
# Coefficients are on the log-mean scale. Canopy slopes apply to the SCALED
# vendor readings (pct - 50) / 10, not to raw percentages.
TRUTH = {
    "intercept": -1.20,
    "canopy_vendor_a": 0.330,   # near-cancels vendor_b: only the sum is identified
    "canopy_vendor_b": -0.295,
    "beer_linear": -0.280,      # with beer_quadratic, gives a U shape
    "beer_quadratic": 0.210,
    "theta": 1.6,               # NB size; variance = mu + mu^2 / theta
}
# Ranger district is an unordered label stored as an integer 1-5. Effects are
# non-monotonic, so a continuous slope averages them to nearly nothing.
DISTRICT_EFFECT = {1: 0.00, 2: 0.62, 3: -0.41, 4: 0.55, 5: -0.15}
DISTRICT_NAME = {
    1: "Mt. Humptulips (reference)",
    2: "Sasquatch Flats",
    3: "Grey Owl Basin",
    4: "Cougar Creek",
    5: "Lake Quinault",
}
SEED = 20261016


def draw_canopy(rng, n, rho=0.972):
    """Two satellite vendors measuring the same canopy cover, in percent.

    Vendor A and Vendor B score the same pixels with different algorithms, so
    their readings are nearly redundant (target correlation rho).
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = np.linalg.cholesky(cov) @ rng.standard_normal((2, n))
    return 50 + 10 * z[0], 50 + 10 * z[1]


def draw_beer(rng, canopy_a, load=0.55):
    """Campground beer sales, z-scored, partly tracking canopy cover.

    Shaded campgrounds sell more beer, so this predictor is entangled with the
    canopy block (correlation ~0.57) as well as having a non-linear effect.
    """
    noise = np.sqrt(1 - load**2) * rng.standard_normal(len(canopy_a))
    return load * (canopy_a - 50) / 10 + noise


def draw_visitor_hundred_hours(rng, canopy_b):
    """Exposure in HUNDREDS of visitor-hours: the denominator every rate needs.

    Lognormal over roughly two orders of magnitude, and mildly correlated with
    canopy, which is what makes omitting the offset actively misleading rather
    than merely inefficient.
    """
    mu_log = np.log(30) + 0.012 * (canopy_b - 50)
    hours = np.exp(rng.normal(mu_log, 0.75, len(canopy_b)))
    return np.round(hours).clip(1, None).astype(int)


def linear_predictor(canopy_a, canopy_b, beer, district, visitor_hundred_hours):
    """Assemble log(mu), including the log-exposure offset with coefficient 1."""
    a_s = (canopy_a - 50) / 10
    b_s = (canopy_b - 50) / 10
    district_term = np.array([DISTRICT_EFFECT[d] for d in district])
    return (
        TRUTH["intercept"]
        + TRUTH["canopy_vendor_a"] * a_s
        + TRUTH["canopy_vendor_b"] * b_s
        + TRUTH["beer_linear"] * beer
        + TRUTH["beer_quadratic"] * beer**2
        + district_term
        + np.log(visitor_hundred_hours)
    )


def draw_sightings(rng, mu, theta=TRUTH["theta"]):
    """Negative binomial counts: overdispersed, heavy-tailed, no zero inflation."""
    return rng.negative_binomial(theta, theta / (theta + mu))


def simulate(n=500, seed=SEED):
    """Build the full survey table."""
    rng = np.random.default_rng(seed)
    canopy_a, canopy_b = draw_canopy(rng, n)
    beer = draw_beer(rng, canopy_a)
    district = rng.integers(1, 6, n)
    visitor_hundred_hours = draw_visitor_hundred_hours(rng, canopy_b)
    mu = np.exp(linear_predictor(canopy_a, canopy_b, beer, district, visitor_hundred_hours))
    return pd.DataFrame({
        "park_month_id": np.arange(1, n + 1),
        "canopy_vendor_a": canopy_a.round(3),
        "canopy_vendor_b": canopy_b.round(3),
        "beer_sales_z": beer.round(4),
        "ranger_district": district,
        "visitor_hundred_hours": visitor_hundred_hours,
        "sightings": draw_sightings(rng, mu),
    })


if __name__ == "__main__":
    df = simulate()
    df.to_csv("cryptid_sightings.csv", index=False)
    rate = df.sightings.sum() / df.visitor_hundred_hours.sum()
    print(f"wrote cryptid_sightings.csv  n={len(df)}")
    print(f"  sightings: mean {df.sightings.mean():.1f}, max {df.sightings.max()}, "
          f"{(df.sightings == 0).mean():.1%} zeros")
    print(f"  variance/mean = {df.sightings.var() / df.sightings.mean():.1f} (Poisson expects 1)")
    print(f"  overall rate = {rate:.2f} sightings per 100 visitor-hours")
