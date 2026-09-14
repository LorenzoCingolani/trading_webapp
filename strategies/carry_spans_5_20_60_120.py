import os
import numpy as np
import pandas as pd

from .carry_spans_utils import (
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
    ann_sharpe,
)

VAR_LOOKBACK = 36
ALPHA = 2.0 / (VAR_LOOKBACK + 1.0)
CARRY_SPANS = [5, 20, 60, 120]


def ewma_var_from_squared(sq: pd.Series, alpha: float) -> pd.Series:
    arr = sq.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    mask = ~np.isnan(arr)
    if not mask.any():
        return pd.Series(out, index=sq.index)

    i0 = int(np.argmax(mask))
    prev = arr[i0]
    out[i0] = prev

    for i in range(i0 + 1, len(arr)):
        x2 = 0.0 if np.isnan(arr[i]) else arr[i]
        prev = alpha * x2 + (1.0 - alpha) * prev
        out[i] = prev

    return pd.Series(out, index=sq.index)


def calibrate_scalar(smoothed_raw: pd.Series, target=10.0) -> float:
    avg_abs = pd.to_numeric(smoothed_raw, errors="coerce").abs().mean()
    if not np.isfinite(avg_abs) or avg_abs <= 0:
        return 1.0
    return float(target / avg_abs)


def run_carry_spans(df_raw, inst_code, distance_years, OUT_DIR):
    # Diagnostic/comparison files go in a subfolder, not directly in OUT_DIR - they don't
    # follow the {instrument}_{model}.csv (Date, capped_forecast, forecast*returns) convention
    # that Validation/PDM/Sharpe scan for in DATA/output_instruments/, so mixing them in there
    # breaks that scan (missing 'Date', wrong columns).
    diag_dir = os.path.join(OUT_DIR, 'carry_spans_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    df = ensure_date_sorted(df_raw).copy()

    px_col   = pick_col(df, ["PX_CLOSE_1D", "px_close_1d", "Close", "close"])
    near_col = pick_col(df, ["near", "NEAR"])
    far_col  = pick_col(df, ["far", "FAR"])
    fx_col   = pick_col(df, ["FX_TO_USD", "fx_to_usd", "FX", "Exchange rate"])
    point_val_col = pick_col(df, ["POINT_VALUE", "point_value", "PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE", "tickValue", "tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE", "tickSize", "tick_size"])

    if px_col is None or near_col is None or far_col is None:
        raise KeyError(f"{inst_code}: need PX_CLOSE_1D, near, far.")
    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or TICK_VALUE/TICK_SIZE.")

    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    near = pd.to_numeric(df[near_col], errors="coerce").astype(float)
    far = pd.to_numeric(df[far_col], errors="coerce").astype(float)
    fx = get_fx_series(df, inst_code) if fx_col else pd.Series(1.0, index=df.index)

    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val = float(pd.to_numeric(df[tick_val_col], errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # Raw carry
    ret_near = near.diff()
    var = ewma_var_from_squared(ret_near.pow(2), ALPHA)
    ann_std = np.sqrt(var) * np.sqrt(TRADING_DAYS)

    price_diff = far - near
    net_expected = price_diff / float(distance_years)
    raw_carry = (net_expected / ann_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Existing sizing scaffold
    st_dev_input = get_input_price_stdev(df)
    price_vol = compute_price_volatility_from_input(st_dev_input, px)
    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    icv_local = block_value * price_vol
    ivv_usd = icv_local * fx

    common = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.NaT,
        px_col: px,
        "near": near,
        "far": far,
        "FX_TO_USD": fx,
        "st_dev_input": st_dev_input,
        "price_volatility": price_vol,
        "POINT_VALUE": point_value,
        "ivv_usd": ivv_usd,
        "raw_carry": raw_carry,
    })

    summary_rows = []
    forecast_map = {}
    cum_series_by_span = {}

    for span in CARRY_SPANS:
        smoothed = raw_carry.ewm(span=span, adjust=False, min_periods=1).mean()
        scalar = calibrate_scalar(smoothed)
        uncapped = smoothed * scalar
        forecast = uncapped.clip(-CAP, CAP)

        forecast_map[f"Carry{span}"] = forecast

        base = common.copy()
        base["carry_span"] = span
        base["smoothed_raw_carry"] = smoothed
        base["forecast_scalar"] = scalar
        base["forecast_uncapped"] = uncapped
        base["forecast"] = forecast

        out = simulate_with_nav_lag(
            base,
            forecast=forecast,
            px=px,
            fx=fx,
            ivv_usd=ivv_usd,
            point_value=point_value,
        )

        turnover, avg_yearly_lots, avg_abs_pos, extra_cols = compute_turnover_exact_cols(out)
        out = pd.concat([out, extra_cols], axis=1)

        m = strategy_metrics_from_usd_pnl(out)
        signal_sh = ann_sharpe(forecast * price_diff.shift(-1))

        row = {
            "instrument": inst_code,
            "rule": f"Carry{span}",
            "span": span,
            "forecast_scalar": scalar,
            "avg_abs_forecast": float(forecast.abs().mean()),
            "pct_capped": float((uncapped.abs() >= CAP).mean()),
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
        summary_rows.append(row)
        cum_series_by_span[f"Carry{span}"] = out["pnl_usd"].fillna(0.0).cumsum().to_numpy()

        out.to_csv(
            os.path.join(diag_dir, f"{inst_code}_CARRY{span}_timeseries.csv"),
            index=False,
        )

    summary = pd.DataFrame(summary_rows).sort_values("span")
    corr = pd.DataFrame(forecast_map).corr()

    summary.to_csv(os.path.join(diag_dir, f"{inst_code}_CARRY_SPAN_COMPARISON.csv"), index=False)
    corr.to_csv(os.path.join(diag_dir, f"{inst_code}_CARRY_SPAN_FORECAST_CORRELATIONS.csv"))

    print("\n" + "=" * 132)
    print(f"{inst_code} — PERFORMANCE FOR DIFFERENT CARRY SPAN LENGTHS")
    print("=" * 132)
    print(
        f"{'Rule':<10}{'Scalar':>10}{'Avg|Fc|':>10}{'%Cap':>9}"
        f"{'AnnRet':>10}{'AnnVol':>10}{'Sharpe':>10}{'Sortino':>10}"
        f"{'Calmar':>10}{'Turnover':>11}{'YrLots':>11}{'Avg|Pos|':>11}"
    )
    print("-" * 132)

    for _, r in summary.iterrows():
        print(
            f"{r['rule']:<10}"
            f"{r['forecast_scalar']:>10.3f}"
            f"{r['avg_abs_forecast']:>10.3f}"
            f"{r['pct_capped']:>8.1%}"
            f"{r['ann_return_usd']:>9.2%}"
            f"{r['ann_vol_usd']:>9.2%}"
            f"{r['executed_sharpe']:>10.3f}"
            f"{r['sortino']:>10.3f}"
            f"{r['calmar']:>10.3f}"
            f"{r['turnover_lots']:>11.2f}"
            f"{r['avg_yearly_lots']:>11.2f}"
            f"{r['avg_abs_pos']:>11.2f}"
        )

    print("=" * 132)
    print("\nFORECAST CORRELATION MATRIX")
    print(corr.round(3).to_string())

    return summary, corr, cum_series_by_span, forecast_map
