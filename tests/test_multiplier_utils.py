import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.multiplier_utils import (
    diversification_multiplier,
    is_positive_semidefinite,
    multiplier_correlation,
    nearest_correlation_matrix,
    validate_multiplier_correlation,
)


def _corr(rows, names=None):
    names = names or [f"r{i}" for i in range(len(rows))]
    return pd.DataFrame(rows, index=names, columns=names, dtype=float)


# Valid (positive semidefinite) 4x4 correlation matrix that stops being PSD once its negative
# entries are floored at zero.
PSD_BREAKING = [
    [1.0, -0.6, 0.8, 0.0],
    [-0.6, 1.0, 0.0, 0.8],
    [0.8, 0.0, 1.0, 0.6],
    [0.0, 0.8, 0.6, 1.0],
]


def test_diagonal_entries_remain_one():
    raw = _corr([[1.0, -0.3, 0.2], [-0.3, 1.0, 0.5], [0.2, 0.5, 1.0]])
    adjusted, _ = multiplier_correlation(raw)
    assert np.allclose(np.diag(adjusted.to_numpy()), 1.0)


def test_diagonal_forced_to_one_even_if_raw_diagonal_is_not():
    raw = _corr([[np.nan, 0.4], [0.4, 0.98]])
    adjusted, _ = multiplier_correlation(raw)
    assert np.allclose(np.diag(adjusted.to_numpy()), 1.0)


def test_positive_correlations_remain_unchanged():
    raw = _corr([[1.0, 0.35, 0.8], [0.35, 1.0, 0.1], [0.8, 0.1, 1.0]])
    adjusted, projected = multiplier_correlation(raw)
    assert not projected
    assert np.allclose(adjusted.to_numpy(), raw.to_numpy())


def test_negative_off_diagonal_correlations_become_zero():
    raw = _corr([[1.0, -0.7, 0.3], [-0.7, 1.0, -0.05], [0.3, -0.05, 1.0]])
    adjusted, _ = multiplier_correlation(raw)
    a = adjusted.to_numpy()
    assert a[0, 1] == 0.0 and a[1, 0] == 0.0
    assert a[1, 2] == 0.0 and a[2, 1] == 0.0
    assert a[0, 2] == pytest.approx(0.3) and a[2, 0] == pytest.approx(0.3)


def test_raw_matrix_is_not_modified():
    raw = _corr([[1.0, -0.7], [-0.7, 1.0]])
    before = raw.copy()
    result = diversification_multiplier(raw, [0.5, 0.5], cap=2.5)
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_frame_equal(result.raw_corr, before)
    assert result.raw_corr.iloc[0, 1] == -0.7
    assert result.multiplier_corr.iloc[0, 1] == 0.0


@pytest.mark.parametrize("rho", [0.0, -0.2, -0.9, -1.0])
def test_two_equal_weight_rules_with_zero_or_negative_correlation_give_sqrt2(rho):
    raw = _corr([[1.0, rho], [rho, 1.0]])
    result = diversification_multiplier(raw, [0.5, 0.5], cap=10.0)
    assert result.uncapped_multiplier == pytest.approx(np.sqrt(2.0))
    assert result.multiplier == pytest.approx(1.4142, abs=1e-4)


def test_perfect_positive_correlation_gives_multiplier_of_one():
    raw = _corr([[1.0, 1.0], [1.0, 1.0]])
    assert diversification_multiplier(raw, [0.5, 0.5], cap=2.5).multiplier == pytest.approx(1.0)


def test_multiplier_cap_still_applies():
    raw = _corr([[1.0, -0.9], [-0.9, 1.0]])
    capped = diversification_multiplier(raw, [0.5, 0.5], cap=1.2)
    assert capped.multiplier == pytest.approx(1.2)
    assert capped.uncapped_multiplier == pytest.approx(np.sqrt(2.0))

    ten_uncorrelated = _corr(np.eye(10))
    result = diversification_multiplier(ten_uncorrelated, np.full(10, 0.1), cap=2.0)
    assert result.uncapped_multiplier == pytest.approx(np.sqrt(10.0))
    assert result.multiplier == pytest.approx(2.0)


def test_non_positive_wcw_gives_unit_multiplier():
    raw = _corr([[1.0, 0.0], [0.0, 1.0]])
    assert diversification_multiplier(raw, [0.0, 0.0], cap=2.5).multiplier == 1.0


def test_undefined_correlation_propagates_nan():
    raw = _corr([[1.0, np.nan], [np.nan, 1.0]])
    assert np.isnan(diversification_multiplier(raw, [0.5, 0.5], cap=2.5).multiplier)


def test_accepts_plain_numpy_array():
    result = diversification_multiplier(np.array([[1.0, -0.5], [-0.5, 1.0]]), [0.5, 0.5], cap=2.5)
    assert result.multiplier == pytest.approx(np.sqrt(2.0))


def _assert_valid_multiplier_matrix(m):
    a = np.asarray(m, dtype=float)
    assert np.allclose(np.diag(a), 1.0)
    assert a.min() >= -1e-10
    assert np.linalg.eigvalsh(a).min() >= -1e-10
    assert np.allclose(a, a.T)


def test_flooring_that_breaks_psd_is_projected_to_a_valid_non_negative_correlation_matrix():
    raw = _corr(PSD_BREAKING)
    assert is_positive_semidefinite(raw.to_numpy())

    floored = np.clip(raw.to_numpy(), 0.0, None)
    np.fill_diagonal(floored, 1.0)
    assert not is_positive_semidefinite(floored)

    adjusted, projected = multiplier_correlation(raw)
    a = adjusted.to_numpy()
    assert projected
    _assert_valid_multiplier_matrix(a)

    # nearer to the floored matrix than the crude alternative (shrinking it towards the identity
    # just enough to be PSD)
    lowest = np.linalg.eigvalsh(floored).min()
    t = -lowest / (1.0 - lowest)
    crude = (1.0 - t) * floored + t * np.eye(4)
    assert np.linalg.norm(a - floored) <= np.linalg.norm(crude - floored) + 1e-9

    result = diversification_multiplier(raw, np.full(4, 0.25), cap=10.0)
    assert result.psd_projected
    assert result.wcw > 0


# RX1's six EWMA-speed correlations (rounded) - the matrix on which the earlier projection
# re-introduced negative entries of about -0.02.
RX1_EWMA_SPEEDS = [
    [1.000, 0.786, 0.351, -0.009, -0.186, -0.264],
    [0.786, 1.000, 0.798, 0.362, 0.005, -0.216],
    [0.351, 0.798, 1.000, 0.811, 0.457, 0.128],
    [-0.009, 0.362, 0.811, 1.000, 0.870, 0.580],
    [-0.186, 0.005, 0.457, 0.870, 1.000, 0.883],
    [-0.264, -0.216, 0.128, 0.580, 0.883, 1.000],
]


def test_rx1_speed_matrix_never_gets_negative_entries_back():
    adjusted, projected = multiplier_correlation(_corr(RX1_EWMA_SPEEDS))
    assert projected
    _assert_valid_multiplier_matrix(adjusted)


def test_random_rank_deficient_matrices_always_satisfy_all_three_conditions():
    rng = np.random.default_rng(11)
    projected_count = 0
    for _ in range(300):
        n = int(rng.integers(3, 9))
        factors = int(rng.integers(1, max(2, n - 1)))
        corr = np.corrcoef(rng.normal(size=(n, factors + 1)) @ rng.normal(size=(factors + 1, 60)))
        adjusted, projected = multiplier_correlation(_corr(corr))
        projected_count += projected
        _assert_valid_multiplier_matrix(adjusted)
        result = diversification_multiplier(_corr(corr), np.full(n, 1.0 / n), cap=100.0)
        assert 1.0 - 1e-9 <= result.uncapped_multiplier <= np.sqrt(n) + 1e-9
    assert projected_count > 10  # the projection path really was exercised


def test_validate_rejects_bad_matrices():
    validate_multiplier_correlation(np.eye(3))
    with pytest.raises(ValueError):
        validate_multiplier_correlation(np.array([[1.0, -0.02], [-0.02, 1.0]]))
    with pytest.raises(ValueError):
        validate_multiplier_correlation(np.array([[1.1, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):  # non-negative, unit diagonal, but not PSD
        validate_multiplier_correlation(np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.9], [0.0, 0.9, 1.0]]))


def test_psd_check_is_skipped_when_flooring_keeps_matrix_valid():
    raw = _corr([[1.0, -0.2, 0.3], [-0.2, 1.0, 0.1], [0.3, 0.1, 1.0]])
    _, projected = multiplier_correlation(raw)
    assert not projected


def test_nearest_correlation_matrix_leaves_valid_matrix_essentially_unchanged():
    valid = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    assert np.allclose(nearest_correlation_matrix(valid), valid, atol=1e-6)


def test_combined_forecast_site_uses_floored_correlation():
    """steps/p2_validation.forecast() must floor negatives: two models with strongly negatively
    correlated returns give sqrt(2), not the raw-correlation multiplier (which would hit the 2.5 cap)."""
    from steps import p2_validation

    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    a = pd.DataFrame({"capped_forecast": np.full(300, 10.0), "forecast_pct_return": x})
    b = pd.DataFrame({"capped_forecast": np.full(300, 10.0), "forecast_pct_return": -0.9 * x + 0.1 * rng.normal(size=300)})
    weights = np.array([0.5, 0.5])

    raw_rho = pd.DataFrame({"a": a["forecast_pct_return"], "b": b["forecast_pct_return"]}).corr().iloc[0, 1]
    assert raw_rho < -0.9
    raw_only_multiplier = min(1.0 / np.sqrt(0.5 * (1 + raw_rho)), p2_validation.FDM_CAP)
    assert raw_only_multiplier == p2_validation.FDM_CAP  # what the pre-change maths would have returned

    final_forecast, m = p2_validation.forecast([a, b], weights)
    assert m == pytest.approx(np.sqrt(2.0))
    assert final_forecast == pytest.approx(np.sqrt(2.0) * 10.0)

    audit = p2_validation.multiplier_audit([a, b], weights, ["a", "b"])
    assert audit.raw_corr.loc["a", "b"] == pytest.approx(raw_rho)  # raw matrix untouched
    assert audit.multiplier_corr.loc["a", "b"] == 0.0
