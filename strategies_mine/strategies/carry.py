import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from . import save
import os
import streamlit as st


# ----------------------------------------------------------------------
# Carver-style carry settings
# ----------------------------------------------------------------------
CARRY_SPANS = [5, 20, 60, 120]
CARRY_SCALER = 30.0
FORECAST_CAP = 20.0
TRADING_DAYS = 256
AUM = 10_000_000
TARGET_VOL = 0.20


def calc(Inst_name, data, exchange_rate=1.0, point_value=50):
    """
    Existing Streamlit entry point.

    The strategy now runs four carry smoothing spans:
        Carry5, Carry20, Carry60, Carry120

    All four use the same raw carry definition, the same scaler (=30),
    the same forecast cap (+/-20), and the same sizing assumptions.
    This isolates the effect of smoothing span.

    A Carver-style comparison table is shown in Streamlit and written to CSV.
    """
    data = data.copy()
    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value

    if ('investing_rate' in data.columns) and ('funding_rate' in data.columns):
        hout = carry_foreign(data)
    elif 'far' in data.columns:
        hout = carry_commodity(data)
    else:
        raise NameError(
            'Input file should include either the column "far" or the columns '
            '"investing_rate" and "funding_rate"'
        )

    os.makedirs(os.path.join('DATA', 'output_instruments'), exist_ok=True)
    os.makedirs(os.path.join('DATA', 'output_plots'), exist_ok=True)

    out_csv = os.path.join(
        'DATA', 'output_instruments', f'{Inst_name}_{hout.name}.csv'
    )
    metrics_csv = os.path.join(
        'DATA', 'output_instruments', f'{Inst_name}_{hout.name}_span_metrics.csv'
    )
    corr_csv = os.path.join(
        'DATA', 'output_instruments', f'{Inst_name}_{hout.name}_forecast_correlations.csv'
    )
    out_plot = os.path.join(
        'DATA', 'output_plots', f'{Inst_name}_{hout.name}.png'
    )

    data.to_csv(out_csv, index=False)
    hout.span_metrics.to_csv(metrics_csv, index=False)
    hout.forecast_correlations.to_csv(corr_csv)

    st.write(f"Saving results to {out_csv}")
    st.write(f"Saving span metrics to {metrics_csv}")
    st.write(f"Saving forecast correlations to {corr_csv}")

    st.header(
        f"Carry Strategy Results for {Inst_name} "
        f"Rows: {data.shape[0]} Columns: {data.shape[1]}"
    )

    st.subheader("Performance for different carry span lengths")
    display_metrics = hout.span_metrics.copy()

    pct_cols = [
        "ann_return",
        "ann_vol",
        "avg_drawdown",
        "max_drawdown",
        "pct_capped",
    ]
    for c in pct_cols:
        if c in display_metrics.columns:
            display_metrics[c] = display_metrics[c].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )

    num_cols = [
        "signal_sharpe",
        "executed_sharpe",
        "sortino",
        "calmar",
        "turnover",
        "forecast_scalar",
        "avg_abs_forecast",
        "avg_yearly_lots",
        "avg_abs_position",
    ]
    for c in num_cols:
        if c in display_metrics.columns:
            display_metrics[c] = display_metrics[c].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else ""
            )

    st.dataframe(display_metrics, use_container_width=True)

    st.subheader("Carry forecast correlation matrix")
    st.dataframe(hout.forecast_correlations.round(3), use_container_width=True)

    with st.expander("Show full carry calculation data", expanded=False):
        st.dataframe(data, use_container_width=True)

    plot_cum_series_multi(
        np.arange(1, data.shape[0] + 1),
        hout.cum_series_by_span,
        out_plot,
        Inst_name,
    )

    return hout


def _ewma_variance_from_returns(returns, alpha):
    """Original recursive EWMA variance logic, made robust to NaNs."""
    sq = pd.to_numeric(returns, errors="coerce") ** 2
    out = pd.Series(np.nan, index=sq.index, dtype=float)

    valid = sq.dropna()
    if valid.empty:
        return out

    first_idx = valid.index[0]
    prev = float(valid.iloc[0])
    out.loc[first_idx] = prev

    started = False
    for idx in sq.index:
        if idx == first_idx:
            started = True
            continue
        if not started:
            continue
        x2 = sq.loc[idx]
        if pd.isna(x2):
            x2 = 0.0
        prev = alpha * float(x2) + (1.0 - alpha) * prev
        out.loc[idx] = prev

    return out


def _performance_from_returns(strategy_returns):
    """
    Metrics on a dimensionless daily return series.
    Used only to compare the four spans consistently.
    """
    r = pd.to_numeric(strategy_returns, errors="coerce").dropna()
    if len(r) < 2:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "executed_sharpe": np.nan,
            "sortino": np.nan,
            "avg_drawdown": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
        }

    ann_mean = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    downside_vol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino = ann_mean / downside_vol if pd.notna(downside_vol) and downside_vol > 0 else np.nan

    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    avg_dd = dd.mean()
    max_dd = dd.min()

    years = len(r) / TRADING_DAYS
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "ann_return": ann_mean,
        "ann_vol": ann_vol,
        "executed_sharpe": sharpe,
        "sortino": sortino,
        "avg_drawdown": avg_dd,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
    }


def _turnover_from_rounded_positions(target_pos):
    """
    Carver-style turnover from actual rounded target positions:

        yearly lots traded = total absolute lots traded / years

        turnover =
            yearly lots traded
            / (2 * average absolute position)

    The factor of 2 converts one-way contract trading into round-turn turnover.

    This fixes the old commodity implementation, which divided an average
    DAILY trade count by the number of years.
    """
    target = pd.to_numeric(target_pos, errors="coerce")
    current = target.shift(1)
    trades = (target - current).abs()

    valid_obs = target.notna().sum()
    years = valid_obs / TRADING_DAYS if valid_obs > 0 else np.nan

    avg_abs_position = current.abs().mean()
    total_lots = trades.sum()

    if pd.isna(years) or years <= 0:
        avg_yearly_lots = np.nan
    else:
        avg_yearly_lots = total_lots / years

    if pd.isna(avg_abs_position) or avg_abs_position <= 0:
        turnover = np.nan
    else:
        turnover = avg_yearly_lots / (2.0 * avg_abs_position)

    return turnover, avg_yearly_lots, avg_abs_position, current, trades


def _build_span_rules(
    data,
    raw_carry,
    pnl_move,
    volatility_scalar,
    prefix="",
):
    """
    Build Carry5 / Carry20 / Carry60 / Carry120.

    Controlled comparison:
      - same raw carry
      - same CARRY_SCALER = 30 for every span
      - same +/-20 cap
      - same volatility sizing
      - only the EWMA smoothing span changes
    """
    metrics = []
    forecasts = {}
    cum_series = {}

    for span in CARRY_SPANS:
        rule = f"Carry{span}"

        smoothed = (
            pd.to_numeric(raw_carry, errors="coerce")
            .ewm(span=span, adjust=False, min_periods=1)
            .mean()
        )

        uncapped = CARRY_SCALER * smoothed
        forecast = uncapped.clip(-FORECAST_CAP, FORECAST_CAP)

        forecasts[rule] = forecast

        data[f'{prefix}raw_carry_span_{span}'] = smoothed
        data[f'{prefix}forecast_{span}'] = forecast

        # Forecast is known at t and earns t+1 P&L.
        signal_pnl = forecast * pd.to_numeric(pnl_move, errors="coerce").shift(-1)

        signal_std = signal_pnl.std(ddof=1)
        signal_sharpe = (
            signal_pnl.mean() * np.sqrt(TRADING_DAYS) / signal_std
            if pd.notna(signal_std) and signal_std > 0
            else np.nan
        )

        subsystem_pos = volatility_scalar * forecast / 10.0
        target_pos = subsystem_pos.round(decimals=0)

        data[f'{prefix}Subsystem_Pos_{span}'] = subsystem_pos
        data[f'{prefix}Target_Pos_{span}'] = target_pos

        (
            turnover,
            avg_yearly_lots,
            avg_abs_position,
            current_pos,
            trades,
        ) = _turnover_from_rounded_positions(target_pos)

        data[f'{prefix}Current_Pos_{span}'] = current_pos
        data[f'{prefix}trades_needed_{span}'] = trades

        # Daily cash P&L from yesterday's rounded position.
        # Using the same underlying near-price move for all spans.
        price_move = pd.to_numeric(data['returns'], errors="coerce")
        point_value = pd.to_numeric(data['point_value'], errors="coerce")
        exchange_rate = pd.to_numeric(data['exchange_rate'], errors="coerce")

        cash_pnl = (
            current_pos
            * price_move
            * point_value
            * exchange_rate
        )

        # Normalize by AUM to obtain daily strategy return.
        strategy_ret = cash_pnl / AUM
        perf = _performance_from_returns(strategy_ret)

        cumulative = cash_pnl.fillna(0.0).cumsum()
        cum_series[rule] = cumulative.to_numpy()

        metrics.append({
            "rule": rule,
            "span": span,
            "forecast_scalar": CARRY_SCALER,
            "avg_abs_forecast": forecast.abs().mean(),
            "pct_capped": (uncapped.abs() >= FORECAST_CAP).mean(),
            "signal_sharpe": signal_sharpe,
            "executed_sharpe": perf["executed_sharpe"],
            "ann_return": perf["ann_return"],
            "ann_vol": perf["ann_vol"],
            "sortino": perf["sortino"],
            "avg_drawdown": perf["avg_drawdown"],
            "max_drawdown": perf["max_drawdown"],
            "calmar": perf["calmar"],
            "hit_rate": perf["hit_rate"],
            "turnover": turnover,
            "avg_yearly_lots": avg_yearly_lots,
            "avg_abs_position": avg_abs_position,
            "observations": int(strategy_ret.notna().sum()),
        })

    metrics_df = pd.DataFrame(metrics)
    forecast_corr = pd.DataFrame(forecasts).corr()

    return metrics_df, forecast_corr, cum_series


def carry_foreign(data):
    hout = CARRYout()
    hout.TH = save.TimeHistory()

    data['near'] = pd.to_numeric(data['near'], errors='coerce')
    data['investing_rate'] = pd.to_numeric(data['investing_rate'], errors='coerce')
    data['funding_rate'] = pd.to_numeric(data['funding_rate'], errors='coerce')

    hout.TH.st_dev = np.array(data['st_dev'])
    hout.px_close = np.array(data['near'])
    hout.TH.start_date = data['Date'].iloc[0] if 'Date' in data.columns else None

    data['stdev_lookback'] = 36
    alpha = 2.0 / (36.0 + 1.0)

    data['returns'] = data['near'].diff()
    data['variance'] = _ewma_variance_from_returns(data['returns'], alpha)
    data['st_dev'] = np.sqrt(data['variance'])
    data['stdev_yearly'] = data['st_dev'] * 16.0

    data['price_diff'] = data['investing_rate'] - data['funding_rate']
    data['raw_carry'] = data['price_diff'] / data['stdev_yearly'].replace(0.0, np.nan)

    data['1%_move'] = data['near'] * 0.01
    data['block_value'] = data['1%_move'] * data['point_value']
    data['ICV'] = data['st_dev'] * data['point_value']
    data['IVV'] = data['ICV'] * data['exchange_rate']
    data['Daily_Cash_Vol_Tgt'] = AUM * TARGET_VOL / 16.0
    data['Volatility_Scalar'] = (
        data['Daily_Cash_Vol_Tgt'] / data['IVV'].replace(0.0, np.nan)
    )

    span_metrics, forecast_corr, cum_series = _build_span_rules(
        data=data,
        raw_carry=data['raw_carry'],
        pnl_move=data['returns'],
        volatility_scalar=data['Volatility_Scalar'],
        prefix="FX_",
    )

    hout.span_metrics = span_metrics
    hout.forecast_correlations = forecast_corr
    hout.cum_series_by_span = cum_series

    # Preserve legacy attributes using Carry20 as the representative rule.
    rep = span_metrics.loc[span_metrics['span'] == 20].iloc[0]
    hout.forecast_ret_sr = rep['signal_sharpe']
    hout.forecast_scalar = rep['forecast_scalar']
    hout.turnover = rep['turnover']
    hout.years = data.shape[0] / TRADING_DAYS
    hout.cum_series = cum_series['Carry20']

    return hout


def carry_commodity(data):
    hout = CARRYout()
    hout.TH = save.TimeHistory()

    data['near'] = pd.to_numeric(data['near'], errors='coerce')
    data['far'] = pd.to_numeric(data['far'], errors='coerce')

    hout.TH.st_dev = np.array(data['st_dev'])
    hout.px_close = np.array(data['near'])
    hout.far = np.array(data['far'])
    hout.TH.start_date = data['Date'].iloc[0] if 'Date' in data.columns else None

    data['stdev_lookback'] = 36
    alpha = 2.0 / (36.0 + 1.0)

    data['returns'] = data['near'].diff()
    data['variance'] = _ewma_variance_from_returns(data['returns'], alpha)
    data['stdev'] = np.sqrt(data['variance'])
    data['stdev_yearly'] = data['stdev'] * 16.0

    data['price_diff'] = data['far'] - data['near']
    data['distance'] = 1.0 / 12.0
    data['net_exp_ret'] = data['price_diff'] / data['distance']
    data['raw_carry'] = (
        data['net_exp_ret'] / data['stdev_yearly'].replace(0.0, np.nan)
    )

    data['1%_move'] = data['near'] * 0.01
    data['block_value'] = data['1%_move'] * data['point_value']

    # Preserve original sizing convention: input st_dev drives ICV.
    input_stdev = pd.to_numeric(data['st_dev'], errors='coerce')
    data['ICV'] = input_stdev * data['point_value']
    data['IVV'] = data['ICV'] * data['exchange_rate']
    data['Daily_Cash_Vol_Tgt'] = AUM * TARGET_VOL / 16.0
    data['Volatility_Scalar'] = (
        data['Daily_Cash_Vol_Tgt'] / data['IVV'].replace(0.0, np.nan)
    )

    span_metrics, forecast_corr, cum_series = _build_span_rules(
        data=data,
        raw_carry=data['raw_carry'],
        pnl_move=data['net_exp_ret'],
        volatility_scalar=data['Volatility_Scalar'],
        prefix="CMDTY_",
    )

    hout.span_metrics = span_metrics
    hout.forecast_correlations = forecast_corr
    hout.cum_series_by_span = cum_series

    # Preserve legacy attributes using Carry20 as representative.
    rep = span_metrics.loc[span_metrics['span'] == 20].iloc[0]
    hout.avg_abs_val_capped_forecast = rep['avg_abs_forecast']
    hout.forecast_ret_sr = rep['signal_sharpe']
    hout.forecast_scalar = rep['forecast_scalar']
    hout.turnover = rep['turnover']
    hout.years = data.shape[0] / TRADING_DAYS
    hout.cum_series = cum_series['Carry20']

    return hout


class CARRYout:
    def __init__(self):
        self.model = 'CARRY'
        self.name = self.model


def plot_cum_series_multi(Days, CumSeriesBySpan, figname, inst_name):
    with st.expander(
        f"Show Carry Cumulative Series Plot for {inst_name}",
        expanded=False
    ):
        fig = plt.figure('Carry cumulative series')
        ax = fig.add_subplot(111)

        for rule, values in CumSeriesBySpan.items():
            ax.plot(Days[:len(values)], values, label=rule)

        ax.set_xlabel('days')
        ax.set_ylabel('P & L')
        ax.legend()
        st.pyplot(fig)

        # Save before closing.
        fig.savefig(figname)
        plt.close(fig)

    return


if __name__ == '__main__':
    # This module is designed to be called by the Streamlit application.
    # Keep the block intentionally minimal so package imports remain valid.
    pass
