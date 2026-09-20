"""Simple price volatility used for ICV (position sizing / turnover) and standardised cost.

Calculated here, in code, from the price series itself - the input file's own 'st_dev' column is
never read. It is the plain rolling standard deviation of the last PRICE_VOL_LOOKBACK DAILY PRICE
CHANGES (price.diff(), not the price levels: a rolling std of levels balloons whenever the price
trends and would overstate daily volatility).

This is deliberately separate from the exponentially weighted moving std that strategies/ewma.py
and strategies/carry.py use to normalise their own forecasts - those are unchanged.
"""
import pandas as pd

PRICE_VOL_LOOKBACK = 20


def simple_price_volatility(px, lookback: int = PRICE_VOL_LOOKBACK) -> pd.Series:
    """Rolling std of the last `lookback` daily price changes, in price points per day. The first
    `lookback` rows are NaN (no estimate yet), which sizing treats as a zero position."""
    changes = pd.to_numeric(pd.Series(px), errors="coerce").diff()
    return changes.rolling(lookback).std()
