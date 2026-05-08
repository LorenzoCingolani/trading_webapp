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

    # ── Cash instruments only need a price column ──────────────────────────────
    # REMOVED: point_val_col, tick_val_col, tick_size_col lookups.
    #   Futures need a contract multiplier (point value) to convert price moves
    #   into P&L.  For stocks, 1 share moves $1 per $1 of price, so the
    #   multiplier is always 1.  No column is needed.
    px_col = pick_col(df, ["PX_CLOSE_1D", "px_close_1d", "Close", "close"])
    if px_col is None:
        raise KeyError(f"{inst_code}: price column not found.")

    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    fx = get_fx_series(df, inst_code)   # still required for non-USD stocks

    # ── point_value is always 1 for cash instruments ───────────────────────────
    # REMOVED: the conditional block that read POINT_VALUE / TICK_VALUE /
    #   TICK_SIZE from the dataframe and computed point_value from them.
    #   For a stock, one share = one unit, so point_value = 1.0 everywhere.
    point_value = 1.0

    # ── Volatility estimates ───────────────────────────────────────────────────
    # These are unchanged; the maths is instrument-agnostic.
    stdev_signal = ewma_stdev_from_net_returns(px, ALPHA)   # for forecast scaling
    st_dev_input = get_input_price_stdev(df)
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)

    # ── Instrument value volatility (IVV) ─────────────────────────────────────
    # With point_value = 1, block_value collapses to one_pct_move, but we keep
    # all intermediate variables so the downstream code (simulate / metrics)
    # receives the same column names it always expected.
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value   # == one_pct_move; kept for clarity
    icv_local    = block_value * price_vol
    ivv_usd      = icv_local * fx

    std_cost = get_standard_cost(df_raw)
    print(f"[{inst_code}] standard_cost={std_cost:.6f}")

    # ── EWMA crossover loop ────────────────────────────────────────────────────
    # Completely unchanged: the signal itself is purely price-derived and has
    # no futures-specific logic.
    for fast, scaler in EWMA_SCALERS.items():
        slow = fast * 4
        ewf = px.ewm(span=fast, adjust=False).mean()
        ews = px.ewm(span=slow, adjust=False).mean()
        raw_cross = ewf - ews
        vol_adj   = raw_cross / stdev_signal.replace(0.0, np.nan)
        forecast  = (vol_adj * scaler).clip(-CAP, CAP)

        base = pd.DataFrame({
            "Date":             df.get("Date", pd.NaT),
            "PX_CLOSE_1D":      px,
            "FX_TO_USD":        fx,
            "stdev_signal":     stdev_signal,
            "st_dev_input":     st_dev_input,
            "price_volatility": price_vol,
            "one_pct_move":     one_pct_move,
            "POINT_VALUE":      point_value,   # always 1.0; kept for schema compat
            "block_value":      block_value,
            "icv_local":        icv_local,
            "ivv_usd":          ivv_usd,
            "forecast":         forecast,
        })

        out = simulate_with_nav_lag(
            base, forecast=forecast, px=px,
            fx=fx, ivv_usd=ivv_usd, point_value=point_value,
        )

        t_over, ayl, aap, extra_cols = compute_turnover_exact_cols(out)
        out = pd.concat([out, extra_cols], axis=1)

        m = strategy_metrics_from_usd_pnl(out)
        m["turnover_lots"]   = t_over
        m["avg_yearly_lots"] = ayl
        m["avg_abs_pos"]     = aap

        raw_sh = raw_signal_sharpe(forecast, px)

        label  = f"{inst_code}_EWMA_{fast}d_{slow}d"
        ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
        mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")

        out.to_csv(ts_csv, index=False)
        pd.DataFrame([{
            "instrument":    inst_code,
            "strategy":      f"EWMA_{fast}/{slow}",
            "raw_sharpe":    raw_sh,
            "standard_cost": std_cost,
            **m,
        }]).to_csv(mt_csv, index=False)

        print(f"[{label}] RawSharpe={raw_sh:.3f} | Sharpe={m['sharpe']:.3f} | "
              f"Turnover={m['turnover_lots']:.2f} | yearly lots={m['avg_yearly_lots']:.2f} | "
              f"avg|pos|={m['avg_abs_pos']:.2f} | Obs={m['obs']}")


# ── Signal-only test  (no sizing, no common_utils) ────────────────────────────
# Drop in any list of closing prices and run:  python 2-ewma-cash.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── swap these prices for any real ticker history you have ─────────────────
    TEST_PRICES = [
        100, 101, 103, 102, 105, 107, 106, 108, 110, 109,
        111, 113, 112, 115, 117, 116, 118, 120, 119, 122,
        124, 123, 126, 128, 127, 130, 132, 131, 134, 136,
        135, 138, 140, 139, 142, 144, 143, 141, 139, 137,
        135, 133, 131, 129, 127, 125, 123, 121, 119, 117,
        115, 117, 119, 121, 123, 125, 127, 129, 131, 133,
        135, 137, 139, 141, 143, 145, 147, 149, 151, 153,
    ]
    # ──────────────────────────────────────────────────────────────────────────

    CAP_LOCAL = 20          # forecast cap (mirrors CAP from common_utils)
    ALPHA_LOCAL = 0.06      # ewma stdev decay  (mirrors ALPHA)

    px = pd.Series(TEST_PRICES, dtype=float)

    # rolling ewma stdev of returns — same formula as ewma_stdev_from_net_returns
    ret   = px.pct_change()
    stdev = ret.ewm(alpha=ALPHA_LOCAL, adjust=False).std()

    print(f"{'EWMA':>10} {'last EWF':>10} {'last EWS':>10} "
          f"{'raw cross':>11} {'forecast':>10}  first non-NaN forecast at row")
    print("─" * 75)

    for fast, scaler in EWMA_SCALERS.items():
        slow      = fast * 4
        ewf       = px.ewm(span=fast, adjust=False).mean()
        ews       = px.ewm(span=slow, adjust=False).mean()
        raw_cross = ewf - ews
        vol_adj   = raw_cross / stdev.replace(0.0, np.nan)
        forecast  = (vol_adj * scaler).clip(-CAP_LOCAL, CAP_LOCAL)

        first_valid = forecast.first_valid_index()
        print(f"{fast:>3}d/{slow:<3}d  "
              f"{ewf.iloc[-1]:>10.3f} {ews.iloc[-1]:>10.3f} "
              f"{raw_cross.iloc[-1]:>11.4f} {forecast.iloc[-1]:>10.3f}  "
              f"row {first_valid}")