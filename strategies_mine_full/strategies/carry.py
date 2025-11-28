
import os
import numpy as np
import pandas as pd
from strategies_mine_full.common.utils import (
    TRADING_DAYS, CAP, ALPHA,
    pick_col, ensure_date_sorted, get_fx_series, get_input_price_stdev,
    ewma_var_from_squared, compute_price_volatility_from_input,
    simulate_with_nav_lag, compute_turnover_exact_cols,
    strategy_metrics_from_usd_pnl, raw_signal_sharpe,
)

CARRY_SCALER = 30.0

def run_carry(df_raw: pd.DataFrame, inst_code: str, distance_years: float, OUT_DIR: str):
    df = ensure_date_sorted(df_raw)

    px_col   = pick_col(df, ["PX_CLOSE_1D","px_close_1d","Close","close"])
    near_col = pick_col(df, ["near","NEAR"])
    far_col  = pick_col(df, ["far","FAR"])
    point_val_col = pick_col(df, ["POINT_VALUE","point_value","PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE","tickValue","tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE","tickSize","tick_size"])
    if None in (px_col, near_col, far_col):
        raise KeyError(f"{inst_code}: need PX_CLOSE_1D/near/far columns.")
    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE).")

    px   = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    near = pd.to_numeric(df[near_col], errors="coerce").astype(float)
    far  = pd.to_numeric(df[far_col],  errors="coerce").astype(float)
    fx   = get_fx_series(df, inst_code)

    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val  = float(pd.to_numeric(df[tick_val_col],  errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # ---------- RAW CARRY (Excel-matched) ----------
    ret_near = near - near.shift(1)            # absolute price change
    sq       = ret_near**2
    var      = ewma_var_from_squared(sq, ALPHA)
    ann_std  = np.sqrt(var) * np.sqrt(TRADING_DAYS)

    price_diff   = far - near                  # ensure FAR minus NEAR
    net_expected = price_diff / float(distance_years)  # distance in years
    raw_carry    = net_expected / pd.Series(ann_std, index=df.index).replace(0.0, np.nan)
    forecast     = (raw_carry * CARRY_SCALER).clip(-CAP, CAP)

    # ---------- SIZING via input st_dev ----------
    st_dev_input = get_input_price_stdev(df)                        # absolute
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)  # %
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value
    icv_local    = block_value * price_vol
    ivv_usd      = icv_local * fx

    base = pd.DataFrame({
        "Date": df.get("Date", pd.NaT),
        "PX_CLOSE_1D": px,
        "near": near, "far": far,
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

    out = simulate_with_nav_lag(base, forecast=forecast, px=px, fx=fx, ivv_usd=ivv_usd, point_value=point_value)

    # raw vs realized metrics
    raw_sh = raw_signal_sharpe(forecast, px)
    m = strategy_metrics_from_usd_pnl(out)

    # Carver-exact turnover
    t_over, ayl, aap, extra_cols = compute_turnover_exact_cols(out)
    out = pd.concat([out, extra_cols], axis=1)
    m["turnover_lots"]   = t_over
    m["avg_yearly_lots"] = ayl
    m["avg_abs_pos"]     = aap

    label  = f"{inst_code}_CARRY"
    ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
    mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")
    out.to_csv(ts_csv, index=False)
    pd.DataFrame([{"instrument":inst_code, "strategy":"CARRY", "raw_sharpe": raw_sh, **m}]).to_csv(mt_csv, index=False)

    print(f"[{label}] RawSharpe={raw_sh:.3f} | Sharpe={m['sharpe']:.3f} | "
          f"Sortino={m['sortino']:.3f} | MaxDD=${m['max_drawdown_usd']:,.0f} | "
          f"Calmar={m['calmar']:.3f} | Turnover={m['turnover_lots']:.2f} | "
          f"yearly lots={m['avg_yearly_lots']:.2f} | avg|pos|={m['avg_abs_pos']:.2f} | Obs={m['obs']}")
    return [(label, ts_csv, mt_csv, m)]
