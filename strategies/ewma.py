import os
import numpy as np
import pandas as pd
import streamlit as st

FORECAST_SCALARS = {2: 10.6, 4: 7.5, 8: 5.3, 16: 3.75, 32: 2.65, 64: 1.87}
CAP = 20.0
TRADING_DAYS = 256
STDEV_LOOKBACK = 36


def _calc_turnover(capped_forecast, px, st_dev, point_value, exchange_rate, aum=10_000_000):
    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    price_volatility = (st_dev / px * 100).round(2)
    icv = price_volatility * block_value
    ivv = icv * exchange_rate
    daily_cash_vol_tgt = aum * 0.2 / 16
    volatility_scalar = daily_cash_vol_tgt / ivv.replace(0.0, np.nan)
    subsystem_pos = volatility_scalar * capped_forecast / 10.0
    target_pos = subsystem_pos.round(0).fillna(0.0)
    current_pos = target_pos.shift()
    trades_needed = target_pos - current_pos

    avg_abs_pos = current_pos.abs().mean()
    years = len(px) / TRADING_DAYS
    if years <= 0 or pd.isna(avg_abs_pos) or avg_abs_pos == 0:
        return np.nan
    trades_needed_yearly = trades_needed.abs().sum() / years
    return trades_needed_yearly / (2.0 * avg_abs_pos)


def calc(Inst_name, data, MAParam, standard_cost, exchange_rate=1.0, point_value=50):
    """
    Fast/slow EWMA price crossover, one span per entry in MAParam - fully vectorized
    (pandas .ewm()), unlike the row-by-row version this replaced.

    Applies a trading-cost filter like EWMA Norm: a speed is only saved and used if its
    max-payable-cost (0.13 / turnover) covers standard_cost, otherwise it's discarded as
    too expensive to trade. Writes DATA/output_instruments/{Inst_name}_EWMA{fast}.csv
    for each speed that passes.

    Returns (summary_df, passed) where passed is {fast: {name, cum_series, avg_abs_val_capped_forecast}}
    for the speeds that passed the cost filter.
    """
    data = data.copy()
    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value

    px = pd.to_numeric(data['PX_CLOSE_1D'], errors='coerce')
    st_dev_input = pd.to_numeric(data['st_dev'], errors='coerce') if 'st_dev' in data.columns else None
    daily_return = px.ffill().pct_change(fill_method=None)
    returns = px.diff().fillna(0.0)

    stdev_decay_alpha = 2.0 / (STDEV_LOOKBACK + 1)
    variance = (returns ** 2).ewm(alpha=stdev_decay_alpha, adjust=False).mean()
    std_dev = np.sqrt(variance)
    turnover_stdev = st_dev_input if st_dev_input is not None else std_dev

    output_folder = os.path.join('DATA', 'output_instruments')
    os.makedirs(output_folder, exist_ok=True)
    # Diagnostics-only file, not a (Date, capped_forecast, forecast*returns) model output -
    # keep it out of output_instruments/ so Validation/PDM/Sharpe's directory scans don't
    # pick it up as a phantom "version" (same reasoning as carry_spans_diagnostics).
    diag_dir = os.path.join(output_folder, 'ewma_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    summary_rows = []
    passed = {}

    for ewma_fast in MAParam:
        if ewma_fast not in FORECAST_SCALARS:
            raise ValueError(f"Invalid value for ewma_fast: {ewma_fast}")
        ewma_slow = ewma_fast * 4
        forecast_scalar = FORECAST_SCALARS[ewma_fast]

        ema_fast = px.ewm(span=ewma_fast, adjust=False).mean()
        ema_slow = px.ewm(span=ewma_slow, adjust=False).mean()
        raw_cross = ema_fast - ema_slow
        vol_adj_crossover = raw_cross / std_dev.replace(0.0, np.nan)
        forecast = vol_adj_crossover * forecast_scalar
        capped_forecast = forecast.clip(-CAP, CAP)

        forecast_returns = capped_forecast * daily_return.shift(-1)
        cum_series = forecast_returns.fillna(0.0).cumsum()

        turnover = _calc_turnover(capped_forecast, px, turnover_stdev, point_value, exchange_rate)

        gross_std = forecast_returns.std()
        gross_mean = forecast_returns.mean()
        gross_sr = (gross_mean * np.sqrt(TRADING_DAYS) / gross_std) if gross_std else np.nan
        if pd.notna(gross_sr) and gross_sr > 0:
            net_sr = gross_sr - (gross_sr * standard_cost)
        elif pd.notna(gross_sr):
            net_sr = gross_sr + (gross_sr * standard_cost)
        else:
            net_sr = np.nan

        max_payable = 0.13 / turnover if pd.notna(turnover) and turnover > 0 else np.nan
        is_good = pd.notna(max_payable) and max_payable >= standard_cost
        avg_abs_forecast = float(capped_forecast.abs().mean())

        summary_rows.append({
            "Speed (fast/slow)": f"{ewma_fast}/{ewma_slow}",
            "Status": "Good_trade" if is_good else "Expensive_discard",
            "Scalar": forecast_scalar,
            "Turnover": turnover,
            "Max Payable": max_payable,
            "Std Cost": standard_cost,
            "Gross SR": gross_sr,
            "Net SR": net_sr,
            "Avg|Forecast|": avg_abs_forecast,
        })

        if is_good:
            name = f"EWMA{ewma_fast:03d}"
            out_df = pd.DataFrame({
                'Date': data['Date'] if 'Date' in data.columns else pd.NaT,
                'PX_CLOSE_1D': px,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'raw_cross': raw_cross,
                'std_dev': std_dev,
                'vol_adj_crossover': vol_adj_crossover,
                'forecast_uncapped': forecast,
                'capped_forecast': capped_forecast,
                'daily_return': daily_return,
                'forecast*returns': forecast_returns,
                'cumulative_performance': cum_series,
                'forecast_scalar': forecast_scalar,
                'turnover': turnover,
                'signal_sharpe': gross_sr,
                'net_sharpe': net_sr,
                'max_payable_cost': max_payable,
                'standard_cost': standard_cost,
                'status': 'Good_trade',
            })
            out_df.to_csv(os.path.join(output_folder, f'{Inst_name}_{name}.csv'), index=False)
            passed[ewma_fast] = {
                'name': name,
                'cum_series': cum_series.to_numpy(),
                'avg_abs_val_capped_forecast': avg_abs_forecast,
            }

    summary_df = pd.DataFrame(summary_rows)

    with st.expander(f"EWMA speed summary for {Inst_name}", expanded=False):
        def highlight_status(row):
            colour = "background-color: #d4edda" if row["Status"] == "Good_trade" else "background-color: #f8d7da"
            return [colour] * len(row)
        st.dataframe(summary_df.style.apply(highlight_status, axis=1), use_container_width=True)
        kept = [r["Speed (fast/slow)"] for r in summary_rows if r["Status"] == "Good_trade"]
        discarded = [r["Speed (fast/slow)"] for r in summary_rows if r["Status"] == "Expensive_discard"]
        st.success(f"Kept: {', '.join(kept) if kept else 'none'}")
        if discarded:
            st.warning(f"Discarded (too expensive to trade): {', '.join(discarded)}")

    summary_df.to_csv(os.path.join(diag_dir, f'{Inst_name}_EWMA_speed_summary.csv'), index=False)

    return summary_df, passed
