import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.volatility import PRICE_VOL_LOOKBACK, simple_price_volatility


def test_lookback_is_20_daily_price_changes():
    assert PRICE_VOL_LOOKBACK == 20
    rng = np.random.default_rng(3)
    px = pd.Series(100 + np.cumsum(rng.normal(size=120)))
    vol = simple_price_volatility(px)
    expected_last = px.diff().iloc[-20:].std()  # sample std (ddof=1) of exactly the last 20 changes
    assert vol.iloc[-1] == pytest.approx(expected_last)
    # no estimate until 20 daily changes exist: row 0 has no change, so first value is row 20
    assert vol.iloc[:20].isna().all()
    assert vol.iloc[20:].notna().all()


def test_uses_price_changes_not_price_levels():
    # a steadily trending price with tiny noise: daily-change vol is tiny, level std is huge
    rng = np.random.default_rng(5)
    px = pd.Series(100 + 1.0 * np.arange(200) + rng.normal(scale=0.05, size=200))
    vol = simple_price_volatility(px).iloc[-1]
    level_std = px.rolling(20).std().iloc[-1]
    assert vol < 0.1
    assert level_std > 5.0
    assert level_std > 50 * vol


def test_ignores_any_st_dev_column_in_the_input_file():
    from p_pages.main_analysis_page import _calc_standard_cost

    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "PX_CLOSE_1D": 100 + np.cumsum(rng.normal(size=80)),
        "BID": 99.9, "ASK": 100.0, "EXECUTION_COST": 0.2, "CLEARING_COST": 0.2, "Standard Cost": 0.002,
    })
    without_column = _calc_standard_cost(df.copy(), 1000.0)
    df["st_dev"] = 12345.0  # absurd input-file value that must not be read
    with_column = _calc_standard_cost(df, 1000.0)
    assert with_column == pytest.approx(without_column)

    expected_sigma = df["PX_CLOSE_1D"].diff().iloc[-20:].std()
    total_one_way = (0.1 / 2) * 1000.0 + 0.2 + 0.2
    assert with_column == pytest.approx(2 * total_one_way / (expected_sigma * 1000.0 * np.sqrt(256)))
