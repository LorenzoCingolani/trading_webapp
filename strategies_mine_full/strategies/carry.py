import os
import numpy as np
import pandas as pd
from math import sqrt

from ..common.utils import (
    TRADING_DAYS,
    CAP,
    pick_col,
    ensure_date_sorted,
    get_fx_series,
    get_input_price_stdev,
    compute_price_volatility_from_input,
    simulate_with_nav_lag,
    compute_turnover_exact_cols,
    strategy_metrics_from_usd_pnl,
    raw_signal_sharpe,
    ann_sharpe,
)

# Carver-style settings
VAR_LOOKBACK = 36
ALPHA = 2.0 / (VAR_LOOKBACK + 1.0)
CARRY_SCALER = 30.0


def ewma_var_from_squared(sq: pd.Series, alpha: float) -> pd.Series:
    """
    EWMA variance on squared near-returns.
    First non-NaN observation is seeded with its own value.
    """
    arr = sq.to_numpy()
    n = len(arr)
    out = np.full(n, np.nan)

    mask = ~np.isnan(arr)
    if not mask.any():
        return pd.Series(out, index=sq.index)

    i0 = np.argmax(mask)
    out[i0] = arr[i0]
    prev = out[i0]

    for i in range(i0 + 1, n):
        x2 = 0.0 if np.isnan(arr[i]) else arr[i]
        prev = alpha * x2 + (1 - alpha) * prev
        out[i] = prev

    return pd.Series(out, index=sq.index)


def run_carry(df_raw: pd.DataFrame, inst_code: str, distance_years: float, OUT_DIR: str):
    """
    Carry strategy:
      - Uses near/far columns
      - distance_years: e.g. 1/12 for monthly, 3/12 for quarterly
      - Uses st_dev from input for ICV/IVV sizing
      - Uses simulate_with_nav_lag() for vol-targeted execution
      - Saves timeseries & metrics CSV into OUT_DIR
      - Prints: signal_sharpe, executed_sharpe, turnover, yearly lots, avg|pos|, obs
    """
    df = ensure_date_sorted(df_raw)

    # ---- detect columns robustly ----
    px_col   = pick_col(df, ["PX_CLOSE_1D", "px_close_1d", "Close", "close"])
    near_col = pick_col(df, ["near", "NEAR"])
    far_col  = pick_col(df, ["far", "FAR"])
    fx_col   = pick_col(df, ["FX_TO_USD", "fx_to_usd", "FX", "Exchange rate"])

    point_val_col = pick_col(df, ["POINT_VALUE", "point_value", "PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE", "tickValue", "tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE", "tickSize", "tick_size"])

    if px_col is None or near_col is None or far_col is None:
        raise KeyError(f"{inst_code}: need PX_CLOSE_1D, near, far columns for carry.")

    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE).")

    px   = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    near = pd.to_numeric(df[near_col], errors="coerce").astype(float)
    far  = pd.to_numeric(df[far_col], errors="coerce").astype(float)

    fx = get_fx_series(df, inst_code) if fx_col else pd.Series(1.0, index=df.index)

    # ---- point value ----
    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val  = float(pd.to_numeric(df[tick_val_col],  errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # ---- carry raw: net_expected_return / ann_std ----
    ret_near = near.diff()
    sq = ret_near ** 2
    var = ewma_var_from_squared(sq, ALPHA)
    ann_std = np.sqrt(var) * np.sqrt(TRADING_DAYS)

    price_diff    = far - near
    net_expected  = price_diff / float(distance_years)
    denom         = pd.Series(ann_std, index=df.index).replace(0.0, np.nan)
    raw_carry     = net_expected / denom
    raw_carry     = raw_carry.replace([np.inf, -np.inf], np.nan)

    forecast = (raw_carry * CARRY_SCALER).clip(-CAP, CAP)

    # ---- sizing scaffold using input st_dev ----
    st_dev_input = get_input_price_stdev(df)  # column like 'st_dev'
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)  # st_dev / px * 100, rounded
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value
    icv_local    = block_value * price_vol       # no /100 (as per your spreadsheet)
    ivv_usd      = icv_local * fx                # convert to USD if fx != 1

    base = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.NaT,
        px_col: px,
        "near": near,
        "far": far,
        "FX_TO_USD": fx,
        "st_dev_input": st_dev_input,
        "price_volatility": price_vol,
        "one_pct_move": one_pct_move,
        "POINT_VALUE": point_value,
        "block_value": block_value,
        "icv_local": icv_local,
        "ivv_usd": ivv_usd,
        "forecast": forecast,
    })

    # ---- simulate with NAV(t-1) vol targeting ----
    out = simulate_with_nav_lag(
        base,
        forecast=forecast,
        px=px,
        fx=fx,
        ivv_usd=ivv_usd,
        point_value=point_value,
    )

    # ---- Carver-exact turnover (rounded positions) ----
    turnover, avg_yearly_lots, avg_abs_pos, extra_cols = compute_turnover_exact_cols(out)
    out = pd.concat([out, extra_cols], axis=1)

    # ---- metrics from PnL ----
    m = strategy_metrics_from_usd_pnl(out)
    # overwrite turnover-related fields with exact version
    m["turnover_lots"]   = turnover
    m["avg_yearly_lots"] = avg_yearly_lots
    m["avg_abs_pos"]     = avg_abs_pos

    # ---- signal (raw) Sharpe ----
    raw_sig_series = forecast * price_diff.shift(-1)
    signal_sh = ann_sharpe(raw_sig_series)

    # ---- pretty console print ----
    print(f"\n[{inst_code}_CARRY]")
    print(f"Signal Sharpe       : {signal_sh:8.3f}")
    print(f"Executed Sharpe     : {m['executed_sharpe']:8.3f}")
    print(f"Sortino             : {m['sortino']:8.3f}")
    print(f"Max DD (USD)        : {m['max_drawdown_usd']:12.0f}")
    print(f"Calmar              : {m['calmar']:8.3f}")
    print(f"Turnover (lots)     : {turnover:8.2f}")
    print(f"Avg yearly lots     : {avg_yearly_lots:8.2f}")
    print(f"Avg |position|      : {avg_abs_pos:8.2f}")
    print(f"Observations        : {m['obs']:8d}")

    # ---- save CSVs ----
    label = f"{inst_code}_CARRY"
    ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
    mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")

    out.to_csv(ts_csv, index=False)

    metrics_row = {
        "instrument": inst_code,
        "strategy": "CARRY",
        "signal_sharpe": signal_sh,
        "executed_sharpe": m["executed_sharpe"],
        "sortino": m["sortino"],
        "ann_return_usd": m["ann_return_usd"],
        "ann_vol_usd": m["ann_vol_usd"],
        "max_drawdown_usd": m["max_drawdown_usd"],
        "calmar": m["calmar"],
        "hit_rate": m["hit_rate"],
        "turnover_lots": turnover,
        "avg_yearly_lots": avg_yearly_lots,
        "avg_abs_pos": avg_abs_pos,
        "obs": m["obs"],
    }
    pd.DataFrame([metrics_row]).to_csv(mt_csv, index=False)

    return out, m
