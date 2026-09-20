import streamlit as st
import os
import re
import pandas as pd
import json
import numpy as np

from steps.multiplier_utils import diversification_multiplier
from steps.volatility import simple_price_volatility
from steps.pipeline_info import show_active_instruments, show_step_explanation, show_source_paths, show_generated_files, show_multiplier_correlation_audit

TRADING_DAYS = 256
SHARPE_RESULTS_FILE = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
SHARPE_RESULTS_CSV_FILE = os.path.join('DATA', 'output_instruments', 'sharpe_results.csv')
NORMAL_TAIL_RATIO = 4.43  # 1st/30th and 99th/70th percentile ratios both equal this for a Gaussian
# Individual EWMA speed versions are excluded from weighting/blending below whenever a
# 'combined' (FDM-blended) version exists for the same instrument - same fix as the
# Validation page's _EWMA_SPEED_RE, for the same reason (double-counting EWMA's contribution).
_EWMA_SPEED_RE = re.compile(r'^EWMA\d{3}$')


def _skew(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 3:
        return np.nan
    return float(r.skew())


def _tail_ratios(returns: pd.Series):
    """Rob Carver's fat-tail ratios, normalized so 1.0 = same as a Gaussian distribution."""
    r = returns.dropna()
    if len(r) < 50:
        return np.nan, np.nan
    p1, p30, p70, p99 = r.quantile([0.01, 0.30, 0.70, 0.99])
    lower_ratio = (p1 / p30) if p30 != 0 else np.nan
    upper_ratio = (p99 / p70) if p70 != 0 else np.nan
    relative_lower = (lower_ratio / NORMAL_TAIL_RATIO) if pd.notna(lower_ratio) else np.nan
    relative_upper = (upper_ratio / NORMAL_TAIL_RATIO) if pd.notna(upper_ratio) else np.nan
    return relative_lower, relative_upper


def _alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, trading_days: int = TRADING_DAYS):
    """OLS of strategy_return = alpha + beta * benchmark_return; alpha annualized (x trading_days)."""
    aligned = pd.concat(
        [strategy_returns, benchmark_returns], axis=1, keys=['strategy', 'benchmark']
    ).dropna()
    if len(aligned) < 30:
        return np.nan, np.nan
    var_bench = aligned['benchmark'].var()
    if not var_bench or pd.isna(var_bench):
        return np.nan, np.nan
    beta = aligned['benchmark'].cov(aligned['strategy']) / var_bench
    alpha_daily = aligned['strategy'].mean() - beta * aligned['benchmark'].mean()
    return alpha_daily * trading_days, beta


def _annualized_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    return float(r.mean() * trading_days) if not r.empty else np.nan


def _annualized_std(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    return float(r.std() * np.sqrt(trading_days)) if len(r) >= 2 else np.nan


def _sortino(returns: pd.Series, trading_days: int = TRADING_DAYS, min_downside_obs: int = 20) -> float:
    r = returns.dropna()
    if len(r) < 3:
        return np.nan
    downside = r[r < 0]
    if len(downside) < min_downside_obs:
        # too few negative-day observations for downside std to be a reliable estimate,
        # rather than an artifact of a tiny sample (e.g. 2 bad days out of 582)
        return np.nan
    downside_std = downside.std()
    if pd.isna(downside_std) or downside_std == 0:
        return np.nan
    return float(r.mean() / downside_std * np.sqrt(trading_days))


def _equity_curve(returns: pd.Series) -> pd.Series:
    # clip at -0.999 so an extreme single-day capped-forecast swing can't send compounded
    # equity to zero/negative and break the drawdown math
    r = returns.fillna(0.0).clip(lower=-0.999)
    return (1.0 + r).cumprod()


def _average_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = _equity_curve(returns)
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.mean())


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = _equity_curve(returns)
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _worst_day(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.min()) if not r.empty else np.nan


def _worst_month(returns: pd.Series, dates: pd.Series) -> float:
    r = returns.fillna(0.0).clip(lower=-0.999)
    parsed_dates = pd.to_datetime(dates, dayfirst=True, errors='coerce')
    valid = parsed_dates.notna()
    if not valid.any():
        return np.nan
    monthly = (1.0 + r[valid]).groupby(parsed_dates[valid].dt.to_period('M')).prod() - 1.0
    return float(monthly.min()) if not monthly.empty else np.nan


def _profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def _net_sharpe(gross_sharpe: float, costs: float, turnover: float) -> float:
    """Carver's SR-units cost adjustment: Costs (SC) is a per-trade cost already expressed in
    Sharpe Ratio units, so multiplying by Turnover (trades/year) annualizes it into the same
    units as Sharpe itself - directly subtractable, no extra volatility scaling needed (unlike
    Cost %, which converts that same drag into %-return terms instead)."""
    if pd.isna(gross_sharpe) or pd.isna(costs) or pd.isna(turnover):
        return np.nan
    return float(gross_sharpe - (costs * turnover))


def _metrics_from_returns(returns: pd.Series, turnover: float, costs: float, dates: pd.Series,
                           benchmark_returns: pd.Series = None, is_dollar_based: bool = False) -> dict:
    """Every METRIC_ROWS value from a single daily-percentage-return series, given a
    precomputed turnover and cost - shared by the main per-strategy loop's logic and the
    combined-forecast (single instrument) rows below.

    is_dollar_based distinguishes two different things that both used to be labelled just
    "Mean Annual Return": a forecast/10 x % price return series (a normalized, notional
    return - reflects the shape/scale of the signal, not real cash P&L) vs. an actual
    position-based $ P&L / capital series (a real portfolio return, e.g. from
    _isolated_instrument_sim or framework_main). Only one of the two keys is populated per
    call; the other is left NaN so every row still has both METRIC_ROWS columns."""
    series = returns.dropna()
    sharpe = (series.mean() / series.std() * np.sqrt(TRADING_DAYS)) if (not series.empty and series.std() > 0) else np.nan
    skew = _skew(returns)
    lower_tail, upper_tail = _tail_ratios(returns)
    if benchmark_returns is not None:
        alpha, beta = _alpha_beta(returns, benchmark_returns)
    else:
        alpha, beta = np.nan, np.nan
    ann_vol = _annualized_std(returns)
    if pd.notna(costs) and pd.notna(turnover) and pd.notna(ann_vol):
        cost_pct = costs * turnover * ann_vol
    else:
        cost_pct = np.nan
    ann_return = _annualized_return(returns)
    return {
        'Normalized Mean Annual Return': np.nan if is_dollar_based else ann_return,
        'Actual Portfolio Return': ann_return if is_dollar_based else np.nan,
        'Costs (SC)': costs,
        'Cost %': cost_pct,
        'Average Drawdown': _average_drawdown(returns),
        'Maximum Drawdown': _max_drawdown(returns),
        'Standard Deviation': ann_vol,
        'Worst Day': _worst_day(returns),
        'Worst Month': _worst_month(returns, dates) if dates is not None else np.nan,
        'Sharpe Ratio (Gross)': sharpe,
        'Sharpe Ratio (Net)': _net_sharpe(sharpe, costs, turnover),
        'Sortino': _sortino(returns),
        'Profit Factor': _profit_factor(returns),
        'Turnover': turnover,
        'Skew': skew,
        'Lower Tail': lower_tail,
        'Upper Tail': upper_tail,
        'Alpha': alpha,
        'Beta': beta,
    }


def _isolated_instrument_sim(px: pd.Series, st_dev: pd.Series, forecast: pd.Series,
                              tick_value: float, tick_size: float, point_value: float,
                              exchange_rate: float, aum: float = 10_000_000) -> dict:
    """Same position-sizing/turnover/carried-forward-P&L math as strategies/ewma.py's
    _calc_turnover() and steps/p5_framework_one_function.py (carried-forward P&L only, no
    execution-slippage guess - see the Forecast page P&L discussion), scoped to one
    instrument at weight=1/PDM=1/fixed-$10M vol target so Turnover stays on the same basis
    as every other row already on this page."""
    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    price_volatility = (st_dev / px * 100).round(2)
    icv = price_volatility * block_value
    ivv = icv * exchange_rate
    daily_cash_vol_tgt = aum * 0.2 / np.sqrt(TRADING_DAYS)
    volatility_scalar = daily_cash_vol_tgt / ivv.replace(0.0, np.nan)
    subsystem_pos = volatility_scalar * forecast / 10.0
    target_pos = subsystem_pos.round(0).fillna(0.0)
    current_pos = target_pos.shift()
    trades_needed = target_pos - current_pos

    avg_abs_pos = current_pos.abs().mean()
    years = len(px) / TRADING_DAYS
    if years <= 0 or pd.isna(avg_abs_pos) or avg_abs_pos == 0:
        turnover = np.nan
    else:
        turnover = (trades_needed.abs().sum() / years) / (2.0 * avg_abs_pos)

    pnl_carried_forward = px.diff() * (tick_value / tick_size) * current_pos
    dollar_pct_return = pnl_carried_forward / aum

    return {
        'target_pos': target_pos,
        'current_pos': current_pos,
        'trades_needed': trades_needed,
        'turnover': turnover,
        'pnl_carried_forward': pnl_carried_forward,
        'dollar_pct_return': dollar_pct_return,
    }


METRIC_ROWS = [
    'Normalized Mean Annual Return', 'Actual Portfolio Return', 'Costs (SC)', 'Cost %',
    'Average Drawdown', 'Maximum Drawdown', 'Standard Deviation', 'Worst Day', 'Worst Month',
    'Sharpe Ratio (Gross)', 'Sharpe Ratio (Net)', 'Sortino', 'Profit Factor', 'Turnover',
    'Skew', 'Lower Tail', 'Upper Tail', 'Alpha', 'Beta',
]

def calculate_sharpe_forecast_returns(csvs_dictionary):
    results = []
    for inst_version, df in csvs_dictionary.items():
        if 'forecast*returns' in df.columns:
            series = df['forecast*returns'].dropna()
            if not series.empty and series.std() > 0:
                sharpe = series.mean() / series.std() * np.sqrt(252)
                # Split inst_version into instrument and version
                if '_' in inst_version:
                    instrument, version = inst_version.split('_', 1)
                    if version == 'results':
                        continue  # Skip results version
                else:
                    instrument, version = inst_version, ''
                results.append({
                    'Instrument': instrument,
                    'Version': version,
                    'Sharpe Ratio': sharpe
                })
    return pd.DataFrame(results)

def run():
    st.title("Performance Metrics")
    st.write("This page calculates the Sharpe ratio for the 'forecast*returns' column in each instrument's dataframe Each Strategy.")
    show_source_paths(["p_pages/sharpe_ratio_page.py :: run() (computed inline, no separate steps/ module)"])

    show_active_instruments()
    show_step_explanation(
        f"For each output file in `DATA/output_instruments/`, computes "
        f"`Sharpe = mean(forecast*returns) / std(forecast*returns) * sqrt({TRADING_DAYS})` "
        f"(annualized over {TRADING_DAYS} trading days/year). You can then set per-strategy-version "
        "weights per instrument to see a blended Sharpe ratio."
    )

    if 'sharpe_started' not in st.session_state:
        st.session_state.sharpe_started = False
    if 'sharpe_done' not in st.session_state:
        st.session_state.sharpe_done = False
    if 'sharpe_results' not in st.session_state:
        st.session_state.sharpe_results = {}

    if st.session_state.sharpe_done:
        st.success("Performance Metrics already completed. Use Run Performance Metrics Again to rerun.")
        results = st.session_state.sharpe_results
        if results:
            st.subheader("Sharpe ratios")
            st.dataframe(results.get("sharpes_df", []))
            st.write("Saved results to DATA/output_instruments/sharpe_results.json")
        show_generated_files([SHARPE_RESULTS_FILE, SHARPE_RESULTS_CSV_FILE], heading="Generated files (Performance Metrics)")
        if st.button("Run Performance Metrics Again", key="rerun_sharpe"):
            st.session_state.sharpe_started = False
            st.session_state.sharpe_done = False
            st.session_state.sharpe_results = {}
            st.rerun()
        return

    if st.button("Run Performance Metrics", key="run_sharpe", type="primary"):
        st.session_state.sharpe_started = True

    if not st.session_state.sharpe_started:
        st.info("Press Run Performance Metrics to calculate ratios and compare strategy versions.")
        return

    input_folder = os.path.join('DATA', 'output_instruments')
    required_cols = {'Date', 'capped_forecast', 'forecast*returns'}
    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if not file.endswith('.csv'):
            continue
        path = os.path.join(input_folder, file)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not required_cols.issubset(df.columns):
            continue
        inst = file.split('_')[0]
        version = file.split('_')[-1].replace('.csv', '')
        csvs_dictionary.setdefault(inst, {})[version] = df

    sharpes = []
    per_instrument_combined = {}  # populated below, used for the true multi-instrument PORTFOLIO rows
    for inst, versions in csvs_dictionary.items():
        for version, df in versions.items():
            if version == "results":
                continue
            if 'forecast*returns' not in df.columns:
                continue
            # forecast*returns can be on a non-percentage, non-i.i.d. scale (e.g. Carry's
            # net_exp_ret basis, which is a smooth curve-implied yield level rather than a daily
            # return - both it and capped_forecast are highly autocorrelated day-to-day, ~0.9+).
            # Feeding that into mean/std*sqrt(256) - a formula that assumes roughly independent
            # daily samples - mechanically inflates Sharpe/Sortino/etc into nonsense (20+). Use the
            # dedicated percentage-return column, which behaves like a genuine daily return
            # (near-zero autocorrelation), for every ratio/distribution-shape metric below. For
            # EWMA/EWMA Norm/Carry Spans, forecast*returns already is a percentage return, so this
            # is a no-op there.
            pct_series = df['forecast_pct_return'] if 'forecast_pct_return' in df.columns else df['forecast*returns']
            series = pct_series.dropna()
            if series.empty or series.std() == 0:
                continue
            sharpe = series.mean() / series.std() * np.sqrt(TRADING_DAYS)
            skew = _skew(pct_series)
            lower_tail, upper_tail = _tail_ratios(pct_series)
            if 'PX_CLOSE_1D' in df.columns:
                benchmark_returns = pd.to_numeric(df['PX_CLOSE_1D'], errors='coerce').pct_change(fill_method=None)
                alpha, beta = _alpha_beta(pct_series, benchmark_returns)
            else:
                alpha, beta = np.nan, np.nan
            costs = float(df['standard_cost'].iloc[0]) if 'standard_cost' in df.columns and len(df) else np.nan
            turnover = float(df['turnover'].iloc[0]) if 'turnover' in df.columns and len(df) else np.nan
            ann_vol = _annualized_std(pct_series)
            if pd.notna(costs) and pd.notna(turnover) and pd.notna(ann_vol):
                sr_cost = costs * turnover
                cost_pct = sr_cost * ann_vol
            else:
                cost_pct = np.nan
            sharpes.append({
                'Instrument': inst,
                'Version': version,
                'Normalized Mean Annual Return': _annualized_return(pct_series),
                'Actual Portfolio Return': np.nan,
                'Costs (SC)': costs,
                'Cost %': cost_pct,
                'Average Drawdown': _average_drawdown(pct_series),
                'Maximum Drawdown': _max_drawdown(pct_series),
                'Standard Deviation': ann_vol,
                'Worst Day': _worst_day(pct_series),
                'Worst Month': _worst_month(pct_series, df['Date']) if 'Date' in df.columns else np.nan,
                'Sharpe Ratio (Gross)': sharpe,
                'Sharpe Ratio (Net)': _net_sharpe(sharpe, costs, turnover),
                'Sortino': _sortino(pct_series),
                'Profit Factor': _profit_factor(pct_series),
                'Turnover': turnover,
                'Skew': skew,
                'Lower Tail': lower_tail,
                'Upper Tail': upper_tail,
                'Alpha': alpha,
                'Beta': beta,
            })

        # Combined-forecast (single instrument) rows: Carry+EWMA blended via the Validation
        # page's FinalForecast, both on a forecast*returns basis and on a real $-P&L basis
        # (isolated single-instrument simulation, weight=1/PDM=1 - see the "single instrument
        # using combined forecast" level of the performance-measures discussion).
        # inst here is truncated to the first underscore-delimited token (e.g. 'AD1' from
        # 'AD1_small_CARRY.csv'), but combinedForecast/, input_instruments/ and
        # control_output.csv all use the full instrument code (e.g. 'AD1_small') - resolve it
        # from control_output.csv's INSTRUMENT column rather than assume 'inst' alone matches.
        control_path = os.path.join('DATA', 'output_instruments', 'control_output.csv')
        full_inst = None
        if os.path.exists(control_path):
            control_df = pd.read_csv(control_path)
            matches = control_df[control_df['INSTRUMENT'].astype(str) == inst]
            if matches.empty:
                matches = control_df[control_df['INSTRUMENT'].astype(str).str.startswith(inst + '_')]
            if not matches.empty:
                full_inst = matches['INSTRUMENT'].iloc[0]

        combined_path = os.path.join('DATA', 'combinedForecast', f'{full_inst}.csv') if full_inst else ''
        if full_inst and os.path.exists(combined_path) and versions:
            try:
                cf_df = pd.read_csv(combined_path)
                cf_df['Date'] = pd.to_datetime(cf_df['Date'], errors='coerce')

                any_df = next(iter(versions.values()))
                px_lookup = any_df[['Date', 'PX_CLOSE_1D']].copy()
                px_lookup['Date'] = pd.to_datetime(px_lookup['Date'], dayfirst=True, errors='coerce')

                input_path = os.path.join('DATA', 'input_instruments', f'{full_inst}.csv')
                raw_df = pd.read_csv(input_path)
                raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=True, errors='coerce')
                # Price volatility for sizing is calculated in code from the full price history
                # (std of the last 20 daily price changes) - the input file's st_dev is not read.
                raw_df['st_dev'] = simple_price_volatility(raw_df['PX_CLOSE_1D'])

                merged = cf_df.merge(px_lookup, on='Date', how='inner') \
                              .merge(raw_df[['Date', 'st_dev']], on='Date', how='left') \
                              .sort_values('Date').reset_index(drop=True)

                control_row = control_df[control_df['INSTRUMENT'] == full_inst]

                if not control_row.empty and not merged.empty:
                    tick_value = float(control_row['TICK_VALUE'].iloc[0])
                    tick_size = float(control_row['TICK_SIZE'].iloc[0])
                    point_value = float(control_row['POINT_VALUE'].iloc[0])
                    exchange_rate = float(control_row['EXCHANGE_RATE'].iloc[0])
                    combined_costs = float(control_row['STANDARD_COST'].iloc[0])

                    px = merged['PX_CLOSE_1D'].astype(float)
                    st_dev_series = merged['st_dev'].astype(float)
                    final_forecast = merged['FinalForecast'].astype(float)
                    benchmark_returns = px.pct_change(fill_method=None)

                    sim = _isolated_instrument_sim(
                        px, st_dev_series, final_forecast, tick_value, tick_size, point_value, exchange_rate
                    )

                    # Same consistently-defined convention as strategies/carry.py's and
                    # strategies/ewma.py's forecast_pct_return columns: yesterday's forecast
                    # (shift(1), no look-ahead) x today's realized % return, divided by 10 to
                    # match the position-sizing convention (subsystem_pos = vol_scalar *
                    # capped_forecast / 10, "forecast of 10" = 1x normal position) - see
                    # strategies/carry.py for the full reasoning. Keeps this series on the same
                    # truthful scale as the per-instrument forecast_pct_return rows it's compared
                    # against; Sharpe/Sortino are unaffected either way (scale-invariant).
                    combined_forecast_returns = (final_forecast.shift(1) / 10) * benchmark_returns
                    row_a = _metrics_from_returns(
                        combined_forecast_returns, sim['turnover'], combined_costs, merged['Date'], benchmark_returns,
                        is_dollar_based=False
                    )
                    row_a['Instrument'] = inst
                    row_a['Version'] = 'Combined_forecast_based'
                    sharpes.append(row_a)

                    row_b = _metrics_from_returns(
                        sim['dollar_pct_return'], sim['turnover'], combined_costs, merged['Date'], benchmark_returns,
                        is_dollar_based=True
                    )
                    row_b['Instrument'] = inst
                    row_b['Version'] = 'Combined_dollar_pnl'
                    sharpes.append(row_b)

                    weight_col = 'INSTRUMENT_WEIGHTS'
                    per_instrument_combined[inst] = {
                        'full_inst': full_inst,
                        'combined_forecast_returns': pd.Series(
                            combined_forecast_returns.values, index=pd.to_datetime(merged['Date'])
                        ),
                        'turnover': sim['turnover'],
                        'costs': combined_costs,
                        'weight': float(control_row[weight_col].iloc[0]) if weight_col in control_row.columns else np.nan,
                    }
            except Exception as ex:
                st.warning(f"Could not compute combined-forecast metrics for {inst}: {ex}")

    # --- True multi-instrument "overall portfolio" rows, same Version names as the single-
    # instrument rows above (Instrument='PORTFOLIO' instead of 'AD1'/'RX1'):
    # - Combined_forecast_based: weight x PDM blend of each instrument's combined_forecast_returns
    # - Combined_dollar_pnl: reuses the Forecast page's own framework_main() (same validated
    #   simulation, not a reimplementation) for the real multi-instrument $ AUM curve
    # These are two independent series, NOT one derived from the other - "gross vs net" on this
    # page refers only to the Sharpe Ratio (Gross)/(Net) columns, computed separately for each. ---
    valid_portfolio_insts = {
        k: v for k, v in per_instrument_combined.items()
        if pd.notna(v['weight']) and v['weight'] > 0
    }
    if len(valid_portfolio_insts) >= 1:
        try:
            weights_map = {k: v['weight'] for k, v in valid_portfolio_insts.items()}

            # PDM: same formula as steps/p3_pdm.py::pdm_main(), computed inline here to avoid
            # that function's heavy Streamlit UI output cluttering this page.
            px_pct_map = {}
            for inst_key, d in valid_portfolio_insts.items():
                raw_df = pd.read_csv(os.path.join('DATA', 'input_instruments', f"{d['full_inst']}.csv"))
                raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=True, errors='coerce')
                px_pct_map[inst_key] = raw_df.set_index('Date')['PX_CLOSE_1D'].astype(float).pct_change(fill_method=None)
            px_pct_df = pd.concat(px_pct_map.values(), axis=1, keys=px_pct_map.keys()).dropna()
            w_vec = np.array([weights_map[k] for k in px_pct_df.columns])
            PDM_UPPER_BOUND = 2
            # Negative off-diagonal correlations are floored at zero for the multiplier only;
            # the raw matrix is what's shown alongside it in the details expander below.
            pdm_result = diversification_multiplier(px_pct_df.corr(), w_vec, PDM_UPPER_BOUND)
            pdm_value = pdm_result.multiplier if pd.notna(pdm_result.multiplier) else 1.0

            # Gross: weight x PDM applied to each instrument's combined_forecast_returns,
            # summed across instruments - same combination structure as real position sizing
            # (weight x PDM), just on the scale-free forecast-return series.
            weighted_series = {
                k: d['combined_forecast_returns'] * weights_map[k] for k, d in valid_portfolio_insts.items()
            }
            weighted_df = pd.concat(weighted_series.values(), axis=1, keys=weighted_series.keys()).dropna()
            portfolio_gross_returns = weighted_df.sum(axis=1) * pdm_value

            total_w = sum(weights_map.values())
            portfolio_turnover = sum(
                weights_map[k] * d['turnover'] for k, d in valid_portfolio_insts.items() if pd.notna(d['turnover'])
            ) / total_w
            # SC_effective = sum(w_i * T_i * SC_i) / sum(w_i * T_i) - turnover-weighted, not just
            # weight-weighted, per the worked example (Section 10): an instrument that trades
            # more (higher turnover) should count for more in the blended standard cost, since
            # its cost drag actually matters more to the portfolio.
            wt_sc_terms = [
                weights_map[k] * d['turnover'] * d['costs']
                for k, d in valid_portfolio_insts.items() if pd.notna(d['turnover']) and pd.notna(d['costs'])
            ]
            wt_terms = [
                weights_map[k] * d['turnover']
                for k, d in valid_portfolio_insts.items() if pd.notna(d['turnover']) and pd.notna(d['costs'])
            ]
            portfolio_costs = sum(wt_sc_terms) / sum(wt_terms) if sum(wt_terms) else np.nan

            row_gross = _metrics_from_returns(
                portfolio_gross_returns, portfolio_turnover, portfolio_costs,
                portfolio_gross_returns.index.to_series(), None, is_dollar_based=False
            )
            row_gross['Instrument'] = 'PORTFOLIO'
            # Same version name as the single-instrument rows above (Combined_forecast_based),
            # not 'Gross_...' - this is an independent series, not "Net_dollar_pnl minus costs".
            # 'Gross'/'Net' on this page means only the Sharpe Ratio (Gross)/(Net) columns.
            row_gross['Version'] = 'Combined_forecast_based'
            sharpes.append(row_gross)

            # Net: the real multi-instrument $ simulation (steps/p5_framework_one_function.py,
            # already validated) run with the actual weights and this same PDM - the true
            # portfolio equity curve, not a reimplementation.
            from steps.p5_framework_one_function import framework_main
            control_rows = control_df[control_df['INSTRUMENT'].isin([d['full_inst'] for d in valid_portfolio_insts.values()])]
            fm_dict = {row['INSTRUMENT']: row.to_dict() for _, row in control_rows.iterrows()}
            raw_csvs_dictionary = {
                d['full_inst']: pd.read_csv(os.path.join('DATA', 'input_instruments', f"{d['full_inst']}.csv"))
                for d in valid_portfolio_insts.values()
            }
            with st.expander("Overall Portfolio $ simulation details (PDM, AUM)", expanded=False):
                st.write(f"PDM used: {pdm_value:.4f}")
                show_multiplier_correlation_audit(pdm_result, title="PDM")
                order_df = framework_main(
                    fm_dict, os.path.join('DATA', 'combinedForecast'), raw_csvs_dictionary,
                    pdm_value, '%d/%m/%Y', 10_000_000, is_markov=False
                )
                st.dataframe(order_df[['AUM', 'total_pnl_today']])

            dollar_pct_returns = order_df['total_pnl_today'] / order_df['AUM'].shift(1)
            row_net = _metrics_from_returns(
                dollar_pct_returns, portfolio_turnover, portfolio_costs,
                order_df.index.to_series(), None, is_dollar_based=True
            )
            row_net['Instrument'] = 'PORTFOLIO'
            row_net['Version'] = 'Combined_dollar_pnl'
            sharpes.append(row_net)
        except Exception as ex:
            st.warning(f"Could not compute overall portfolio metrics: {ex}")

    sharpes_df = pd.DataFrame(sharpes)

    st.subheader("Performance Metrics by Strategy")
    with st.expander("What each metric means", expanded=False):
        st.markdown(
            "Every metric below is computed from `forecast_pct_return` - a consistently-defined "
            "percentage-return series saved by every strategy (Carry and EWMA alike): "
            "`(yesterday's capped_forecast / 10) x today's realized % price return`, no "
            "look-ahead. The /10 matches the position-sizing convention (subsystem_pos = "
            "vol_scalar x capped_forecast / 10 - 'forecast of 10' = 1x normal position), so "
            "Mean Annual Return/Std Dev/Drawdowns/Worst Day-Month reflect what a realistically-"
            "sized position would actually earn, not a fictitious ~10x-overlevered version of "
            "the strategy. Correlation/FDM/Sharpe/Sortino are scale-invariant, so the /10 makes "
            "no difference to those - it only affects the absolute-scale metrics below. "
            "`forecast*returns` itself stays on whatever scale each strategy "
            "deliberately saves it on (for Carry, that's the net_exp_ret/raw-price scale, not a "
            "percentage - see the Carry Sharpe discussion) and is shown as-is in the Returns Time "
            "Series section below, but it isn't used for these ratio/statistics metrics: mixing "
            "Carry's net_exp_ret-scale series with EWMA's percentage-scale series would produce "
            "incompatible units and a false combined Sharpe ratio (for Carry specifically, its raw "
            "forecast*returns is also a smooth, highly autocorrelated curve-implied yield level, "
            "~0.9 day-to-day autocorrelation, rather than an independent daily observation - "
            "feeding that into `mean/std*sqrt(256)`, which assumes roughly independent daily "
            "samples, mechanically inflates Sharpe/Sortino into meaningless numbers, 20+).\n\n"
            "- **Normalized Mean Annual Return** - average daily `forecast_pct_return` × 256 trading "
            "days: a notional return built from the (forecast/10 x % price return) signal series, "
            "not real cash P&L. Populated on the forecast-based rows (single EWMA speed, Carry "
            "standalone, `Combined_forecast_based`) - blank on dollar-based rows.\n"
            "- **Actual Portfolio Return** - average daily $ P&L / capital × 256 trading days: the "
            "real return a position-based simulation would have produced (`_isolated_instrument_sim` "
            "or the full multi-instrument `framework_main` run). Populated only on `Combined_dollar_pnl` "
            "rows - blank on forecast-based rows.\n"
            "- **Costs (SC)** - Carver's standardised cost: the instrument's per-trade trading cost, "
            "expressed directly in Sharpe Ratio units (blank if that strategy doesn't save this).\n"
            "- **Cost %** - that cost converted into annualized percentage-return terms: "
            "`(Costs (SC) × Turnover) × Standard Deviation`. Since Sharpe = return / vol, multiplying "
            "the SR-unit cost by turnover (cost drag per year) and then by volatility converts it back "
            "into \"how many percentage points of annual return this strategy loses to trading costs\" - "
            "comparable across instruments regardless of their volatility.\n"
            "- **Average Drawdown** - mean depth of the underwater curve on a compounded equity "
            "curve built from daily % P&L (`equity / running-peak equity - 1`), across the whole "
            "history.\n"
            "- **Maximum Drawdown** - the single worst peak-to-trough decline on that same "
            "compounded equity curve.\n"
            "- **Standard Deviation** - annualized volatility of daily % P&L.\n"
            "- **Worst Day** - the single worst daily % P&L observation.\n"
            "- **Worst Month** - daily % P&L compounded within each calendar month, then the worst "
            "month across the whole history.\n"
            "- **Sharpe Ratio (Gross)** - `mean / std * sqrt(256)` of the daily percentage return "
            "(see note above) - no trading cost deducted.\n"
            "- **Sharpe Ratio (Net)** - Gross minus `Costs (SC) x Turnover`. Costs (SC) is already "
            "expressed in Sharpe Ratio units per trade, so multiplying by Turnover (trades/year) "
            "annualizes it into the same units as Sharpe itself - directly subtractable, unlike "
            "Cost % which converts that same drag into %-return terms instead.\n"
            "- **Sortino** - like Sharpe, but only penalizes downside volatility (negative-day std). "
            "Blank if there are fewer than 20 negative-return days - too small a sample for the "
            "downside std to be a reliable estimate rather than a fluke of 1-2 unlucky days.\n"
            "- **Profit Factor** - sum of gains on winning days divided by the sum of losses on "
            "losing days. Above 1 means gains outweigh losses; below 1 means the reverse.\n"
            "- **Turnover** - annualized position turnover (blank if that strategy doesn't save this).\n"
            "- **Skew** - shape of the return distribution. Positive skew (many small losses, "
            "occasional big wins) is often desirable for convex/crisis-alpha strategies.\n"
            "- **Lower Tail** - Rob Carver's fat-tail ratio: (1st percentile / 30th percentile of "
            f"returns) / {NORMAL_TAIL_RATIO}, where {NORMAL_TAIL_RATIO} is what a Gaussian distribution "
            "gives. Below 1 = thinner (safer) left tail than normal; above 1 = fatter/more extreme "
            "crash risk than normal - lower is better.\n"
            "- **Upper Tail** - same idea at the top: (99th percentile / 70th percentile) / "
            f"{NORMAL_TAIL_RATIO}. Above 1 = fatter upside tail than normal, which can be useful "
            "(e.g. trend-following convexity) - higher can be desirable.\n"
            "- **Alpha** - daily alpha × 256 trading days, from regressing `forecast*returns` "
            "on the instrument's own buy-and-hold return (`strategy_return = alpha + beta × "
            "benchmark_return`). Positive alpha means the strategy added value beyond just holding "
            "the instrument.\n"
            "- **Beta** - slope of that same regression: how much the strategy's daily returns move "
            "with simply holding the instrument. Near 0 = little relationship (more diversifying); "
            "near 1 = behaves like buy-and-hold; negative = tends to move opposite the instrument."
        )

    if not sharpes_df.empty:
        strategy_table = sharpes_df.copy()
        strategy_table['strategy name'] = strategy_table['Instrument'] + ' ' + strategy_table['Version']
        display_df = strategy_table.set_index('strategy name')[METRIC_ROWS].T
        display_df.index.name = None
        st.table(display_df.style.format(precision=4, na_rep="No data"))
    else:
        st.write("No strategy output files found.")

    instruments = list(csvs_dictionary.keys())
    selected_inst = st.selectbox("Select Instrument", instruments)

    all_versions = [v for v in csvs_dictionary[selected_inst].keys() if v != "results"]
    has_combined_ewma = 'combined' in all_versions
    versions = [v for v in all_versions if not (has_combined_ewma and _EWMA_SPEED_RE.match(v))]
    n_versions = len(versions)
    default_weight = 1.0 / n_versions if n_versions > 0 else 0.0

    st.subheader(f"Set Weights for {selected_inst} Versions (sum ≤ 1.0)")
    weights = []
    total_weight = 0.0
    for version in versions:
        weight = st.number_input(
            f"Weight for {version}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.01,
            key=f"{selected_inst}_{version}_weight"
        )
        weights.append(weight)
        total_weight += weight

    if total_weight > 1.0:
        st.warning("Total weight exceeds 1.0. Weights will be normalized.")
        weights = [w / total_weight for w in weights]
        total_weight = sum(weights)

    st.subheader("Weighted Sharpe Ratios")
    weighted_sharpes = []
    sum_weighted_sharpe = 0.0
    for version, weight in zip(versions, weights):
        sharpe_row = sharpes_df[(sharpes_df['Instrument'] == selected_inst) & (sharpes_df['Version'] == version)]
        if not sharpe_row.empty:
            sharpe = sharpe_row['Sharpe Ratio (Gross)'].values[0]
            weighted = sharpe * weight
            weighted_sharpes.append({'Version': version, 'Weight': weight, 'Sharpe Ratio (Gross)': sharpe, 'Weighted Sharpe': weighted})
            sum_weighted_sharpe += weighted

    st.dataframe(pd.DataFrame(weighted_sharpes))
    st.write(f"**Sum of Weighted Sharpe Ratios:** {sum_weighted_sharpe:.4f}")

    st.subheader(f"Returns Time Series for {selected_inst} (first 10 rows per version)")
    tabs = st.tabs(versions)
    for i, version in enumerate(versions):
        df = csvs_dictionary[selected_inst][version]
        with tabs[i]:
            if 'forecast*returns' in df.columns:
                st.write(f"**{selected_inst} - {version}**")
                st.dataframe(df['forecast*returns'].head(10))

                daily = df['forecast*returns']
                chart_dates = None
                if 'Date' in df.columns:
                    parsed_dates = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    if parsed_dates.notna().any():
                        chart_dates = parsed_dates

                st.caption("Daily forecast*returns")
                st.line_chart(pd.Series(daily.values, index=chart_dates) if chart_dates is not None else daily)

                st.caption("Cumulative performance (running total of forecast*returns)")
                cumulative = daily.fillna(0.0).cumsum()
                st.line_chart(pd.Series(cumulative.values, index=chart_dates) if chart_dates is not None else cumulative)

    st.download_button(
        label="Download Sharpe Ratios CSV",
        data=sharpes_df.to_csv(index=False).encode('utf-8'),
        file_name='sharpe_ratios_forecast_returns.csv',
        mime='text/csv'
    )

    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    sharpes_dict_to_save = sharpes_df.to_dict(orient='records')
    with open(sharpe_results_path, 'w') as f:
        json.dump(sharpes_dict_to_save, f, indent=2)

    sharpes_df.to_csv(SHARPE_RESULTS_CSV_FILE, index=False)

    st.session_state.sharpe_results = {
        "sharpes_df": sharpes_df.to_dict(orient="records"),
        "output_path": sharpe_results_path,
        "csv_output_path": SHARPE_RESULTS_CSV_FILE
    }

    returns_list = []
    for version in versions:
        df = csvs_dictionary[selected_inst][version]
        # Use forecast_pct_return when available (Carry's raw forecast*returns is deliberately
        # on the net_exp_ret scale - highly autocorrelated, not an independent daily return -
        # feeding it into mean/std*sqrt(256) here would reproduce the same inflated-Sharpe bug
        # already fixed in the main per-strategy table above).
        pct_col = 'forecast_pct_return' if 'forecast_pct_return' in df.columns else 'forecast*returns'
        if pct_col in df.columns:
            returns_list.append(df[pct_col].reset_index(drop=True))
    if returns_list:
        returns_matrix = pd.concat(returns_list, axis=1).dropna()
        weights_arr = np.array(weights)
        weights_input = st.text_input(
            "Enter weights as comma-separated values",
            value=",".join(map(str, weights_arr)),
            key=f"{selected_inst}_weights_text_input",
        )
        try:
            parsed = np.array([float(w) for w in weights_input.split(",")])
            if len(parsed) != returns_matrix.shape[1]:
                st.warning(
                    f"Entered {len(parsed)} weight(s) but {selected_inst} has "
                    f"{returns_matrix.shape[1]} version(s) ({', '.join(versions)}) - "
                    "falling back to the equal/adjusted weights above."
                )
                weights_arr = weights_arr / weights_arr.sum()
            else:
                weights_arr = parsed / parsed.sum()
        except Exception:
            weights_arr = np.array(weights)
            weights_arr = weights_arr / weights_arr.sum()

        st.write(f"**Using Weights (normalized):** {weights_arr}")
        portfolio_returns = returns_matrix.dot(weights_arr)
        if portfolio_returns.std() > 0:
            portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        else:
            portfolio_sharpe = np.nan
        st.subheader("Overall Portfolio Sharpe Ratio")
        st.write(f"**Portfolio Sharpe Ratio (weighted): {portfolio_sharpe:.4f}**")
        st.caption("Daily weighted portfolio_returns")
        st.line_chart(portfolio_returns, use_container_width=True)
        st.caption("Cumulative performance (running total of weighted portfolio_returns)")
        st.line_chart(portfolio_returns.fillna(0.0).cumsum(), use_container_width=True)
    else:
        st.write("No valid return series found for selected versions.")

    st.session_state.sharpe_done = True
    st.session_state.sharpe_started = False

    show_generated_files([SHARPE_RESULTS_FILE, SHARPE_RESULTS_CSV_FILE], heading="Generated files (Sharpe Ratio)")

