import numpy as np
import pandas as pd

from steps.returns import pct_returns, pct_returns_of


def test_matches_pct_change_without_held_price():
    px = pd.Series([100.0, 101.0, 99.5, 102.0])
    pd.testing.assert_series_equal(pct_returns(px), px.pct_change())


def test_old_files_without_current_price_unchanged():
    # Existing input files: near can differ from PX_CLOSE_1D around rolls; keep plain pct_change
    df = pd.DataFrame({'PX_CLOSE_1D': [100.0, 101.0, 104.0], 'near': [100.0, 98.0, 99.0]})
    pd.testing.assert_series_equal(pct_returns_of(df), df['PX_CLOSE_1D'].pct_change())
    pd.testing.assert_series_equal(pct_returns_of(df, fallback_col='near'), df['near'].pct_change())


def test_simple_roll_example():
    # Contract A 100, 101, 102 (roll day: B = 105), then B 106, 104.
    # Back-adjusted (+3 up to the roll day): 103, 104, 105, 106, 104.
    df = pd.DataFrame({
        'PX_CLOSE_1D': [103.0, 104.0, 105.0, 106.0, 104.0],
        'near': [100.0, 101.0, 102.0, 106.0, 104.0],
        'far': [None, None, 105.0, None, None],
        'current_price': [100.0, 101.0, 105.0, 106.0, 104.0],
    })
    r = pct_returns_of(df).to_numpy()
    np.testing.assert_allclose(r[1:], [1 / 100, 1 / 101, 1 / 105, -2 / 106])


def test_negative_adjusted_price_keeps_sign():
    # Adjusted levels far back can be negative; the return must still follow the real move.
    df = pd.DataFrame({'PX_CLOSE_1D': [-44.0, -43.5], 'current_price': [71.0, 71.5]})
    assert pct_returns_of(df).iloc[1] > 0
