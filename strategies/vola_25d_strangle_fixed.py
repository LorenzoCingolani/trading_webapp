"""
25-delta strangle backtest, sell-at-cycle-start / roll-on-3rd-Friday.

Replaces the row-by-row recompute in vola.py, which re-derived strikes and
lot size from THAT DAY's spot/ATR on every row (i.e. re-sold the strangle
daily instead of once per monthly cycle), used a fixed +/-2% moneyness band
instead of solving for the actual 25-delta strike, and had a sign bug on the
put leg (norm.ppf(0.25) reused from the call side, when the put convention
needs norm.ppf(1-0.25)) that produced negative put prices for realistic
inputs.

Here, strikes and lot size are solved ONCE at the start of each cycle
(inverting Black-Scholes delta) and frozen for the life of the cycle.
Roll convention: same-day roll -- the old position settles at intrinsic
value using the 3rd Friday's close, and the new position is sold using that
same close.

Expected input CSV columns: Date (dd/mm/yyyy), spx_close, IVOL, rf_rate,
BEST_DIV_YLD, atr.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import norm

MULTIPLIER = 100
TARGET_DELTA = 0.25   # magnitude, applies to both legs
AUM = 1_000_000
RISK_PCT_OF_AUM = 0.01  # 1% of AUM per 1x ATR move, per lot


def get_third_friday(month: int, year: int) -> date:
    d = date(year, month, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def get_third_friday_after(d: date) -> date:
    """Expiry that governs the cycle a position opened on date d belongs to.
    If d IS the current month's 3rd Friday, that's an expiry/roll day, so the
    position opened on d expires the FOLLOWING month (same-day roll)."""
    tf = get_third_friday(d.month, d.year)
    if d >= tf:
        nxt = d + timedelta(days=32 - d.day)
        tf = get_third_friday(nxt.month, nxt.year)
    return tf


def strike_for_delta(S, sigma, T, r, q, target_delta, is_call):
    """Invert Black-Scholes spot delta for K.
    Call delta = N(d1); Put delta = N(d1) - 1 (magnitude = target_delta)."""
    d1 = norm.ppf(target_delta) if is_call else norm.ppf(1 - target_delta)
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r - q + 0.5 * sigma ** 2) * T)


def bs_price(S, K, sigma, T, r, q, is_call):
    if T <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def run_backtest(vol_data: pd.DataFrame) -> pd.DataFrame:
    vol_data = vol_data.copy()
    vol_data["Date_d"] = pd.to_datetime(vol_data["Date"], format="%d/%m/%Y").dt.date
    vol_data = vol_data.sort_values("Date_d").reset_index(drop=True)

    records = []
    current_expiry = None
    call_K = put_K = lots = None
    prev_call_mark = prev_put_mark = 0.0

    for _, row in vol_data.iterrows():
        d = row["Date_d"]
        S = row["spx_close"]
        sigma = row["IVOL"]
        r = row["rf_rate"]
        q = row["BEST_DIV_YLD"]
        atr = row["atr"]

        daily_pnl = 0.0
        rolled = False
        settle_call = settle_put = np.nan

        if current_expiry is not None and d >= current_expiry:
            settle_call = max(S - call_K, 0.0)
            settle_put = max(put_K - S, 0.0)
            daily_pnl += (prev_call_mark - settle_call) * lots * MULTIPLIER
            daily_pnl += (prev_put_mark - settle_put) * lots * MULTIPLIER
            current_expiry = None

        entry_premium = np.nan
        if current_expiry is None:
            current_expiry = get_third_friday_after(d)
            T0 = (current_expiry - d).days / 360
            call_K = strike_for_delta(S, sigma, T0, r, q, TARGET_DELTA, is_call=True)
            put_K = strike_for_delta(S, sigma, T0, r, q, TARGET_DELTA, is_call=False)
            lots = round((AUM * RISK_PCT_OF_AUM) / (atr * MULTIPLIER))
            prev_call_mark = bs_price(S, call_K, sigma, T0, r, q, True)
            prev_put_mark = bs_price(S, put_K, sigma, T0, r, q, False)
            entry_premium = (prev_call_mark + prev_put_mark) * MULTIPLIER * lots
            rolled = True
        else:
            T = (current_expiry - d).days / 360
            call_mark = bs_price(S, call_K, sigma, T, r, q, True)
            put_mark = bs_price(S, put_K, sigma, T, r, q, False)
            daily_pnl += (prev_call_mark - call_mark) * lots * MULTIPLIER
            daily_pnl += (prev_put_mark - put_mark) * lots * MULTIPLIER
            prev_call_mark, prev_put_mark = call_mark, put_mark

        records.append({
            "Date": d,
            "spx_close": S,
            "expiry": current_expiry,
            "call_K": call_K,
            "put_K": put_K,
            "lots": lots,
            "call_mark": prev_call_mark,
            "put_mark": prev_put_mark,
            "rolled_today": rolled,
            "entry_premium": entry_premium,
            "settle_call": settle_call,
            "settle_put": settle_put,
            "pnl_daily": daily_pnl,
        })

    out = pd.DataFrame(records)
    out["pnl_cumulative"] = out["pnl_daily"].cumsum()
    out["NAV"] = AUM + out["pnl_cumulative"]
    return out


def sharpe(pnl_daily: pd.Series) -> float:
    return round(pnl_daily.mean() * np.sqrt(252) / pnl_daily.std(), 4)


def max_drawdown(nav: pd.Series):
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return dd.min(), dd
