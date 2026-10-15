"""Synthetic fixtures matching the documented structure of the Dryad deposit.

THESE ARE NOT THE PAPER'S DATA. The deposit (doi:10.5061/dryad.bzkh189pt,
Part_1.xlsx / Part_2.xlsx / statistical_analyses.R) is behind an OAuth bearer
token and was not retrievable, so the column names below are inferred from the
Methods section, not read off the files. They exist so the pipeline can be
executed and tested end to end; every number they produce is a property of
this generator, not a finding about dogs.

Adapt `COLUMN_MAP` in the notebook once the real workbooks are in hand.
"""

import numpy as np
import pandas as pd

PAIRINGS = ["joy_baseline", "stress_baseline", "joy_stress"]
PRESENTATIONS = ["A1", "A2", "A3", "A4", "B"]
CONDITIONS = ["joy", "stress", "blank"]


def simulate_part1(n_dogs=43, seed=7):
    """Habituation-dishabituation: up to 3 pairings x 5 presentations per dog."""
    rng = np.random.default_rng(seed)
    rows = []
    for dog in range(1, n_dogs + 1):
        dog_re = rng.normal(0, 0.35)
        period = rng.choice([1, 2, 3], p=[14 / 43, 17 / 43, 12 / 43])
        pairings = rng.choice(PAIRINGS, size=rng.integers(2, 4), replace=False)
        for pairing in pairings:
            for pres in PRESENTATIONS:
                # declining investigation across A1-A4, partial recovery at B
                base = {"A1": 1.9, "A2": 1.5, "A3": 1.25, "A4": 1.2, "B": 1.6}[pres]
                enhanced = int(rng.random() < 0.15)
                mu = base + dog_re + 0.1 * enhanced
                rows.append({
                    "dog": f"D{dog:02d}", "pairing": pairing, "presentation": pres,
                    "investigation_s": float(np.expm1(rng.normal(mu, 0.45))),
                    "side": rng.choice(["left", "right"]),
                    "owner_enhanced": enhanced, "period": int(period),
                })
    df = pd.DataFrame(rows)
    df["investigation_s"] = df["investigation_s"].clip(lower=0)
    return df


def add_preference_index(part1):
    """B / (A4 + B) per dog per pairing, the dual-box phase preference."""
    wide = part1.pivot_table(index=["dog", "pairing"], columns="presentation",
                             values="investigation_s")
    out = wide.reset_index()[["dog", "pairing"]].copy()
    tot = wide["A4"] + wide["B"]
    out["pref_index"] = (wide["B"] / tot.replace(0, np.nan)).values
    return out.dropna(subset=["pref_index"])


def simulate_part2(n_per_condition=(9, 9, 6), n_bins=12, seed=11):
    """Part 2: 12 five-second sample points per dog, one condition per dog."""
    rng = np.random.default_rng(seed)
    rows = []
    dog_id = 0
    for cond, n in zip(CONDITIONS, n_per_condition):
        for _ in range(n):
            dog_id += 1
            dog = f"P2_{dog_id:02d}"
            sex = rng.choice(["M", "F"])
            p_owner = {"joy": 0.45, "stress": 0.60, "blank": 0.35}[cond]
            p_owner = float(np.clip(p_owner + rng.normal(0, 0.10), 0.02, 0.98))
            for b in range(1, n_bins + 1):
                rows.append({
                    "dog": dog, "condition": cond, "sex": sex, "bin": b,
                    "near_owner": int(rng.random() < p_owner),
                    "near_stranger": int(rng.random() < 0.25),
                    "stress_behaviour": int(rng.random() < (0.30 if cond != "blank" else 0.18)),
                    "joy_behaviour": int(rng.random() < (0.22 if cond == "blank" else 0.15)),
                    "posture": rng.choice(["upright", "sit", "prone"], p=[0.6, 0.1, 0.3]),
                })
    return pd.DataFrame(rows)


def simulate_reliability(n_trials=25, seed=5):
    """Double-coded subsample: two coders, continuous durations + one binary item."""
    rng = np.random.default_rng(seed)
    truth = rng.lognormal(1.0, 0.5, n_trials)
    coder_a = truth * rng.normal(1.0, 0.08, n_trials)
    coder_b = truth * rng.normal(1.0, 0.12, n_trials) + rng.normal(0, 0.2, n_trials)
    item_truth = rng.random(n_trials) < 0.4
    flip_a = rng.random(n_trials) < 0.08
    flip_b = rng.random(n_trials) < 0.15
    return pd.DataFrame({
        "trial": np.arange(1, n_trials + 1),
        "duration_coder_a": coder_a.clip(0), "duration_coder_b": coder_b.clip(0),
        "lipslick_coder_a": (item_truth ^ flip_a).astype(int),
        "lipslick_coder_b": (item_truth ^ flip_b).astype(int),
    })
