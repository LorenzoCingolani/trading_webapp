import os
import numpy as np
import pandas as pd
from common_utils import (
    TRADING_DAYS, CAP, ALPHA,
    pick_col, ensure_date_sorted, get_fx_series, get_input_price_stdev,
    ewma_stdev_from_net_returns, compute_price_volatility_from_input,
    simulate_with_nav_lag, compute_turnover_exact_cols,
    strategy_metrics_from_usd_pnl, raw_signal_sharpe, get_standard_cost,
)

# slow = 4 × fast
EWMA_SCALERS = {2:10.6, 4:7.5, 8:5.3, 16:3.75, 32:2.65, 64:1.87}

def run_ewma(df_raw: pd.DataFrame, inst_code: str, OUT_DIR: str):
    df = ensure_date_sorted(df_raw)

    px_col  = pick_col(df, ["PX_CLOSE_1D","px_close_1d","Close","close"])
    point_val_col = pick_col(df, ["POINT_VALUE","point_value","PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE","tickValue","tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE","tickSize","tick_size"])
    if px_col is None: raise KeyError(f"{inst_code}: price column not found.")
    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE).")

    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    fx = get_fx_series(df, inst_code)

    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val  = float(pd.to_numeric(df[tick_val_col],  errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # signal stdev for vol-adjusting crossover
    stdev_signal = ewma_stdev_from_net_returns(px, ALPHA)

    # sizing stdev via input st_dev
    st_dev_input = get_input_price_stdev(df)
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value
    icv_local    = block_value * price_vol
    ivv_usd      = icv_local * fx

    std_cost = get_standard_cost(df_raw)
    print(f"[{inst_code}] standard_cost={std_cost:.6f}")

    for fast, scaler in EWMA_SCALERS.items():
        slow = fast * 4
        ewf = px.ewm(span=fast, adjust=False).mean()
        ews = px.ewm(span=slow, adjust=False).mean()
        raw_cross = ewf - ews
        vol_adj   = raw_cross / stdev_signal.replace(0.0, np.nan)
        forecast  = (vol_adj * scaler).clip(-CAP, CAP)

        base = pd.DataFrame({
            "Date": df.get("Date", pd.NaT),
            "PX_CLOSE_1D": px,
            "FX_TO_USD": fx,
            "stdev_signal": stdev_signal,
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

        # Carver-exact turnover
        t_over, ayl, aap, extra_cols = compute_turnover_exact_cols(out)
        out = pd.concat([out, extra_cols], axis=1)

        # realized metrics
        m = strategy_metrics_from_usd_pnl(out)
        m["turnover_lots"]   = t_over
        m["avg_yearly_lots"] = ayl
        m["avg_abs_pos"]     = aap

        # raw (signal) Sharpe
        raw_sh = raw_signal_sharpe(forecast, px)

        label  = f"{inst_code}_EWMA_{fast}d_{slow}d"
        ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
        mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")

        out.to_csv(ts_csv, index=False)
        pd.DataFrame([{"instrument":inst_code,"strategy":f"EWMA_{fast}/{slow}",
                       "raw_sharpe": raw_sh, "standard_cost": std_cost, **m}]).to_csv(mt_csv, index=False)

        print(f"[{label}] RawSharpe={raw_sh:.3f} | Sharpe={m['sharpe']:.3f} | "
              f"Turnover={m['turnover_lots']:.2f} | yearly lots={m['avg_yearly_lots']:.2f} | "
              f"avg|pos|={m['avg_abs_pos']:.2f} | Obs={m['obs']}")
