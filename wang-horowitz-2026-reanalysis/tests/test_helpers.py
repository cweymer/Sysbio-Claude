"""Validation of the statistical helpers against independently known values."""

import sys, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis"))
import stats_helpers as H


def test_h_max_ceilings():
    """The two ceilings that distinguish a dog-level from a bin-level Part 2 test."""
    assert H.h_max([9, 9, 6]) == pytest.approx(20.25, abs=0.01)
    assert H.h_max([12, 12, 12]) == pytest.approx(31.14, abs=0.01)


def test_published_part2_H_exceeds_dog_ceiling():
    """Two reported H values cannot have come from 9/9/6 dogs."""
    ceiling = H.h_max([9, 9, 6])
    assert 23.98 > ceiling and 23.11 > ceiling


def test_min_detectable_d_matches_review_figures():
    """Sensitivity figures quoted in the statistical review, to 2 dp."""
    assert H.min_detectable_d(9, 9) == pytest.approx(1.41, abs=0.01)
    assert H.min_detectable_d(9, 6) == pytest.approx(1.60, abs=0.01)
    assert H.min_detectable_d(40) == pytest.approx(0.45, abs=0.01)
    assert H.min_detectable_d(20) == pytest.approx(0.66, abs=0.01)


def test_bh_fdr_monotone_and_bounded():
    p = [0.001, 0.008, 0.013, 0.042, 0.045, 0.3, 0.8]
    out = H.bh_fdr(p)
    assert (out.p_bh <= 1).all()
    assert (out.p_bh >= out.p_raw - 1e-12).all()
    assert out.p_bh.is_monotonic_increasing


def test_perm_paired_recovers_known_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 40)
    y = x - 1.5 + rng.normal(0, 0.5, 40)
    res = H.perm_paired(x, y, n_perm=5000, rng=rng)
    assert res["p_perm"] < 0.01
    assert res["mean_diff"] == pytest.approx(1.5, abs=0.3)


def test_icc_detects_systematic_offset_that_pearson_misses():
    """The reason ICC, not Pearson r, is the right duration statistic."""
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 50)
    b = a + 5.0                       # perfectly correlated, badly disagreeing
    from scipy import stats as st
    assert st.pearsonr(a, b).statistic == pytest.approx(1.0, abs=1e-9)
    assert H.icc21(np.column_stack([a, b])) < 0.8


def test_rank_biserial_sign_and_bounds():
    x = np.array([5.0, 6, 7, 8]); y = np.array([1.0, 2, 3, 4])
    assert H.rank_biserial_paired(x, y) == pytest.approx(1.0)
    assert H.rank_biserial_paired(y, x) == pytest.approx(-1.0)
