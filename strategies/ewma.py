import os
import numpy as np
import pandas as pd
import streamlit as st

FORECAST_SCALARS = {2: 10.6, 4: 7.5, 8: 5.3, 16: 3.75, 32: 2.65, 64: 1.87}
CAP = 20.0
TRADING_DAYS = 256
STDEV_LOOKBACK = 36


def _calc_turnover(capped_forecast, px, st_dev, point_value, exchange_rate, aum=10_000_000):
    """Returns a dict with every intermediate step, plus the final 'turnover' ratio, so the
    caller can save them as columns for validation against a manual/spreadsheet calc."""
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
        trades_needed_yearly = np.nan
        turnover = np.nan
    else:
        trades_needed_yearly = trades_needed.abs().sum() / years
        turnover = trades_needed_yearly / (2.0 * avg_abs_pos)

    return {
        'daily_cash_vol_tgt': daily_cash_vol_tgt,
        'aum': aum,
        'one_pct_move': one_pct_move,
        'block_value': block_value,
        'price_volatility_pct': price_volatility,
        'icv': icv,
        'ivv': ivv,
        'volatility_scalar': volatility_scalar,
        'subsystem_pos': subsystem_pos,
        'target_pos': target_pos,
        'current_pos': current_pos,
        'trades_needed': trades_needed,
        'avg_abs_pos': avg_abs_pos,
        'years': years,
        'trades_needed_yearly': trades_needed_yearly,
        'turnover': turnover,
    }


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
    daily_return = px.ffill().pct_change(fill_method=None)
    # Leave day 1 as NaN (no prior close to diff against) instead of filling it to 0.0 -
    # filling it would seed the variance EWMA with a fake zero-return day 1, damping every
    # subsequent value by a spurious factor of alpha that (with a 36-day lookback, alpha~0.054)
    # takes dozens of observations to decay out. ewm(adjust=False) skips the leading NaN and
    # seeds correctly at the first real day-over-day return, matching a manual/spreadsheet calc.
    returns = px.diff()

    stdev_decay_alpha = 2.0 / (STDEV_LOOKBACK + 1)
    variance = (returns ** 2).ewm(alpha=stdev_decay_alpha, adjust=False).mean()
    std_dev = np.sqrt(variance)
    # Sizing (turnover/position-sizing) uses the input file's own pre-computed st_dev for now
    # (easier to validate directly against it), falling back to the EWMA std_dev above if a
    # given input file doesn't have one.
    if 'st_dev_input_file' in data.columns:
        turnover_stdev = pd.to_numeric(data['st_dev_input_file'], errors='coerce')
    else:
        turnover_stdev = std_dev

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

        turnover_calc = _calc_turnover(capped_forecast, px, turnover_stdev, point_value, exchange_rate)
        turnover = turnover_calc['turnover']

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
                'variance': variance,
                'std_dev': std_dev,
                'vol_adj_crossover': vol_adj_crossover,
                'forecast_uncapped': forecast,
                'capped_forecast': capped_forecast,
                'daily_return': daily_return,
                'forecast*returns': forecast_returns,
                'cumulative_performance': cum_series,
                'forecast_scalar': forecast_scalar,
                'turnover_stdev': turnover_stdev,
                'exchange_rate': exchange_rate,
                'point_value': point_value,
                'one_pct_move': turnover_calc['one_pct_move'],
                'block_value': turnover_calc['block_value'],
                'price_volatility_pct': turnover_calc['price_volatility_pct'],
                'icv': turnover_calc['icv'],
                'ivv': turnover_calc['ivv'],
                'daily_cash_vol_tgt': turnover_calc['daily_cash_vol_tgt'],
                'aum': turnover_calc['aum'],
                'volatility_scalar': turnover_calc['volatility_scalar'],
                'subsystem_pos': turnover_calc['subsystem_pos'],
                'target_pos': turnover_calc['target_pos'],
                'current_pos': turnover_calc['current_pos'],
                'trades_needed': turnover_calc['trades_needed'],
                'avg_abs_pos': turnover_calc['avg_abs_pos'],
                'years': turnover_calc['years'],
                'trades_needed_yearly': turnover_calc['trades_needed_yearly'],
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
                'capped_forecast': capped_forecast.to_numpy(),
                'forecast*returns': forecast_returns.to_numpy(),
            }

    summary_df = pd.DataFrame(summary_rows)

    if passed:
        speeds = sorted(passed.keys())
        n = len(speeds)
        # Carver's FDM (Forecast Diversification Multiplier): correlate the passing speeds'
        # daily forecast*returns, weight them (equal-weighted here), and scale the blended
        # forecast up by M = 1/sqrt(wTCw) so diversification across less-correlated speeds
        # earns a bigger position-sizing multiplier, capped at 2.0.
        forecast_ret_mat = pd.DataFrame({f: passed[f]['forecast*returns'] for f in speeds})
        capped_fc_mat = pd.DataFrame({f: passed[f]['capped_forecast'] for f in speeds})
        corr_mat = forecast_ret_mat.corr()

        weights = np.ones(n) / n
        wcw = float(np.dot(weights.T, np.dot(corr_mat.fillna(0.0).to_numpy(), weights)))
        fdm = min(1.0 / np.sqrt(wcw), 2.0) if wcw > 0 else 1.0

        # min_count=1 so a row where every speed is NaN (day 1, before any crossover is
        # defined) correctly stays NaN - pandas' default sum(skipna=True) would otherwise
        # silently treat "nothing to sum" as 0.0, not "undefined".
        weighted_sum = capped_fc_mat.mul(weights, axis=1).sum(axis=1, min_count=1)
        combined_forecast = (fdm * weighted_sum).clip(-CAP, CAP)
        combined_forecast_returns = combined_forecast * daily_return.shift(-1)
        combined_cum = combined_forecast_returns.fillna(0.0).cumsum()

        combined_df = pd.DataFrame({
            'Date': data['Date'] if 'Date' in data.columns else pd.NaT,
            'PX_CLOSE_1D': px,
            'daily_return': daily_return,
            # 'capped_forecast' (not 'combined_forecast') is deliberate: this is the exact
            # column name Sharpe Ratio/Validation's file scans require to recognize a valid
            # model file. Duplicated below as 'combined_forecast' just so it's unambiguous at
            # a glance next to the per-speed forecast_EWMA0xx columns.
            'capped_forecast': combined_forecast,
            'combined_forecast': combined_forecast,
            'forecast*returns': combined_forecast_returns,
            'cumulative_performance': combined_cum,
            'fdm': fdm,
            'n_speeds_combined': n,
            'speeds_combined': ', '.join(f'EWMA{f:03d}' for f in speeds),
        })
        for i, f in enumerate(speeds):
            combined_df[f'weight_EWMA{f:03d}'] = weights[i]
            combined_df[f'forecast_EWMA{f:03d}'] = capped_fc_mat[f].to_numpy()
        combined_df.to_csv(os.path.join(output_folder, f'{Inst_name}_EWMA_combined.csv'), index=False)
        corr_mat.to_csv(os.path.join(diag_dir, f'{Inst_name}_EWMA_speed_correlation.csv'))

        with st.expander(f"EWMA speed correlation & FDM for {Inst_name}", expanded=False):
            st.write(f"Speeds combined: {', '.join(f'EWMA{f:03d}' for f in speeds)} (equal-weighted, {1.0/n:.4f} each)")
            st.write("Correlation matrix (of each speed's daily forecast*returns):")
            st.dataframe(corr_mat)
            st.write(f"FDM = 1/sqrt(wTCw) = **{fdm:.4f}** (capped at 2.0)")

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
