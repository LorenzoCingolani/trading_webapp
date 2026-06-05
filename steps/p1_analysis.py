import numpy as np
import os
import pandas as pd
import streamlit as st
import time
from collections import namedtuple
from typing import Dict, Iterable

from strategies import break_model, carry, stoch
from strategies_mine import ewma_no_tick


from strategies import stochastic_breakout as breakout

TRADING_DAYS = 256
EWMA_NORM_RULES = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.1, 32: 2.79, 64: 1.91}
EWMA_NORM_FDM_BY_RULE_COUNT = {
    1: 1.0,
    2: 1.02,
    3: 1.03,
    4: 1.23,
    5: 1.25,
    6: 1.27,
}
FORECAST_CAP = 20.0


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _progress_text(label: str, done: int, total: int, start_time: float) -> str:
    elapsed = time.time() - start_time
    if done > 0 and total > 0:
        eta = elapsed / done * (total - done)
        return (
            f"{label}: {done}/{total} | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"ETA {_format_seconds(eta)}"
        )
    return f"{label}: {done}/{total} | elapsed {_format_seconds(elapsed)} | ETA calculating..."


def _get_ewma_norm_fdm(rule_count: int) -> float:
    if rule_count <= 0:
        return 1.0
    return EWMA_NORM_FDM_BY_RULE_COUNT.get(rule_count, 1.27)


def _calculate_forecast_turnover(
    data: pd.DataFrame,
    forecast: pd.Series,
    exchange_rate: float,
    point_value: float,
    aum: float = 10000000,
) -> float:
    px = pd.to_numeric(data["PX_CLOSE_1D"], errors="coerce")
    if "st_dev" in data.columns:
        st_dev = pd.to_numeric(data["st_dev"], errors="coerce")
    else:
        st_dev = px.rolling(20).std()

    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    price_volatility = st_dev / px * 100.0
    icv = price_volatility * block_value
    ivv = (icv * exchange_rate).replace(0.0, np.nan)
    daily_cash_vol_target = aum * 0.2 / 16.0
    volatility_scalar = daily_cash_vol_target / ivv
    subsystem_pos = volatility_scalar * forecast / 10.0
    target_pos = subsystem_pos.round().fillna(0.0)
    current_pos = target_pos.shift().fillna(0.0)
    trades_needed = target_pos - current_pos

    avg_abs_pos = current_pos.abs().mean()
    if pd.isna(avg_abs_pos) or avg_abs_pos == 0:
        return np.nan

    years = len(data) / TRADING_DAYS
    if years <= 0:
        return np.nan

    avg_yearly_lots = trades_needed.abs().sum() / years
    return avg_yearly_lots / (2.0 * avg_abs_pos)


def _compute_ewma_norm(
    data: pd.DataFrame,
    standard_cost: float = 0.0,
    exchange_rate: float = 1.0,
    point_value: float = 1.0,
) -> pd.DataFrame:
    df = data.copy()
    standard_cost = pd.to_numeric(pd.Series([standard_cost]), errors="coerce").iloc[0]
    exchange_rate = pd.to_numeric(pd.Series([exchange_rate]), errors="coerce").fillna(1.0).iloc[0]
    point_value = pd.to_numeric(pd.Series([point_value]), errors="coerce").fillna(1.0).iloc[0]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    px = pd.to_numeric(df["PX_CLOSE_1D"], errors="coerce")
    daily_returns = px.pct_change()
    annual_vol = daily_returns.ewm(span=32, adjust=False).std() * np.sqrt(TRADING_DAYS)
    ten_year_vol = annual_vol.rolling(TRADING_DAYS * 10, min_periods=1).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annual_vol
    daily_price_risk = (weighted_vol / np.sqrt(TRADING_DAYS)) * px
    daily_price_risk = daily_price_risk.replace(0.0, np.nan)

    normalised_returns = 100.0 * (px.diff() / daily_price_risk)
    normalised_price = normalised_returns.fillna(0.0).cumsum()

    passed_forecasts = []
    cost_rows = []
    for fast_span, scalar in EWMA_NORM_RULES.items():
        slow_span = fast_span * 4
        ewmac = (
            normalised_price.ewm(span=fast_span, adjust=False).mean()
            - normalised_price.ewm(span=slow_span, adjust=False).mean()
        )
        forecast = (ewmac * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        turnover = _calculate_forecast_turnover(df, forecast, exchange_rate, point_value)
        max_payable = 0.13 / turnover if pd.notna(turnover) and turnover > 0 else np.nan
        cost_pass = (
            pd.isna(standard_cost)
            or standard_cost <= 0
            or (pd.notna(max_payable) and max_payable >= standard_cost)
        )

        df[f"ewma_norm_{fast_span}_forecast"] = forecast
        df[f"ewma_norm_{fast_span}_turnover"] = turnover
        df[f"ewma_norm_{fast_span}_max_payable"] = max_payable
        df[f"ewma_norm_{fast_span}_cost_pass"] = cost_pass
        cost_rows.append(
            {
                "fast_span": fast_span,
                "slow_span": slow_span,
                "scalar": scalar,
                "turnover": turnover,
                "max_payable": max_payable,
                "standard_cost": standard_cost,
                "cost_pass": cost_pass,
            }
        )
        if cost_pass:
            passed_forecasts.append(forecast)

    if not passed_forecasts:
        combined_forecast = pd.Series(np.nan, index=df.index)
    else:
        fdm = _get_ewma_norm_fdm(len(passed_forecasts))
        combined_forecast = (pd.concat(passed_forecasts, axis=1).mean(axis=1) * fdm).clip(
        -FORECAST_CAP, FORECAST_CAP
        )

    df.attrs["ewma_norm_cost_filter"] = cost_rows
    df["normalised_price_ewma_norm"] = normalised_price
    df["capped_forecast"] = combined_forecast
    df["forecast*returns"] = combined_forecast.shift(1) * daily_returns
    df["cum_series"] = df["forecast*returns"].fillna(0.0).cumsum()
    return df


def main_analysis(
                  framework_dict: Dict[str, Dict[str, float]],
                  csvs_dictionary: Dict[str, pd.DataFrame],
                  selected_strategies: Iterable[str] | None = None) -> None:
    """
    Run financial strategy models (EWMA, Breakout, Carry, etc.) on instruments' data.
    Display results using Streamlit.
    """
    if selected_strategies is None:
        selected_strategies = ["EWMA", "CARRY"]
    selected_strategies = {str(s).upper() for s in selected_strategies}

    ModelsList = []
    if "EWMA" in selected_strategies:
        ModelsList.extend(['ewma01', 'ewma02', 'ewma03', 'ewma04'])
    if "BREAKOUT" in selected_strategies:
        ModelsList.append('breakout')
    if "CARRY" in selected_strategies:
        ModelsList.append('carry')
    if "EWMA_NORM" in selected_strategies:
        ModelsList.append('ewma_norm')

    MAParam = [2, 4, 8, 16]
    BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]

    StrategyResult = namedtuple("StrategyResult", ["name", "cum_series", "avg_abs_val_capped_forecast"])

    analysis_start = time.time()
    instrument_items = list(csvs_dictionary.items())
    total_instruments = len(instrument_items)
    progress_bar = st.progress(0.0)
    progress_status = st.empty()

    for inst_idx, (ins_name, data) in enumerate(instrument_items, start=1):
        progress_status.info(_progress_text("Main analysis instruments", inst_idx - 1, total_instruments, analysis_start))
        st.subheader(f'Instrument name: {ins_name}')

        if ins_name not in framework_dict:
            st.warning(f"No parameters found for {ins_name}")
            progress_bar.progress(inst_idx / total_instruments if total_instruments else 1.0)
            continue

        params = framework_dict[ins_name]
        Inst_name = params['INSTRUMENT']
        Standard_Cost = params['STANDARD_COST']
        exchange_rate = params['EXCHANGE_RATE']
        point_value = params['POINT_VALUE']

        st.write(f"--- Analyzing {Inst_name} ---")

        if Inst_name not in csvs_dictionary:
            st.warning(f'No data found for {Inst_name}. Available keys: {list(csvs_dictionary.keys())}')
            continue

        data = csvs_dictionary[Inst_name].copy()

        StrategyName = []
        CumList = []
        AvgCapForecastList = []
        AvgCapForecastDict = {}

        if 'breakout' in ModelsList:
            st.warning("Breakout was selected, but this analysis path does not currently write Breakout model output files.")

        if 'ewma01' in ModelsList and 'ewma03' in ModelsList:
            st.info('Running EWMA Strategy')
            ewma_df = ewma_no_tick.compute_all_ewma(
                data,
                ewma_no_tick.FORECAST_SCALERS,
                cap=ewma_no_tick.CAP,
            )
            ResList = []
            passed_ewma_strategies = []

            for (fast, slow), _ in ewma_no_tick.FORECAST_SCALERS.items():
                forecast_col = f"ewma_{fast}d_{slow}d_forecast"
                pnl_col = f"ewma_{fast}d_{slow}d_fcastxret"
                if forecast_col not in ewma_df.columns or pnl_col not in ewma_df.columns:
                    continue

                forecast_series = ewma_df[forecast_col]
                if forecast_series.dropna().empty:
                    continue

                res_name = f"EWMA{fast:03d}"
                res = StrategyResult(
                    name=res_name,
                    cum_series=ewma_df[pnl_col].fillna(0.0).cumsum().to_numpy(),
                    avg_abs_val_capped_forecast=float(forecast_series.abs().mean()),
                )
                ResList.append(res)
                passed_ewma_strategies.append(res_name)

                if fast in MAParam:
                    output_folder = os.path.join('DATA', 'output_instruments')
                    os.makedirs(output_folder, exist_ok=True)
                    ewma_output = ewma_df.copy()
                    ewma_output['capped_forecast'] = ewma_output[forecast_col]
                    ewma_output['forecast*returns'] = ewma_output[pnl_col]
                    ewma_output_path = os.path.join(output_folder, f'{Inst_name}_{res_name}.csv')
                    ewma_output.to_csv(ewma_output_path, index=False)

            st.write(f"Passed EWMA strategies: {passed_ewma_strategies}")

            for param in MAParam:
                for res in ResList:
                    if res.name == f"EWMA{param:03d}":
                        if res.name in passed_ewma_strategies:
                            StrategyName.append(res.name)
                            CumList.append(res.cum_series)
                            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
                            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        if 'ewma_norm' in ModelsList:
            st.info('Running EWMA Norm Strategy')
            ewma_norm_output = _compute_ewma_norm(
                data,
                standard_cost=Standard_Cost,
                exchange_rate=exchange_rate,
                point_value=point_value,
            )
            cost_filter_df = pd.DataFrame(ewma_norm_output.attrs.get("ewma_norm_cost_filter", []))
            if not cost_filter_df.empty:
                st.write("EWMA Norm cost filter")
                st.dataframe(cost_filter_df)
            forecast_series = ewma_norm_output['capped_forecast']
            if forecast_series.dropna().empty:
                st.warning(f"EWMA Norm generated no forecast values for {Inst_name}; all spans failed the cost filter.")
            else:
                output_folder = os.path.join('DATA', 'output_instruments')
                os.makedirs(output_folder, exist_ok=True)
                ewma_norm_output_path = os.path.join(output_folder, f'{Inst_name}_EWMA_NORM.csv')
                ewma_norm_output.to_csv(ewma_norm_output_path, index=False)

                res = StrategyResult(
                    name="EWMA_NORM",
                    cum_series=ewma_norm_output['cum_series'].fillna(0.0).to_numpy(),
                    avg_abs_val_capped_forecast=float(forecast_series.abs().mean()),
                )
                StrategyName.append(res.name)
                CumList.append(res.cum_series)
                AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
                AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        carry_has_point_value = pd.notna(point_value) and point_value != 0
        carry_has_tick = (
            'TICK_VALUE' in data.columns and 'TICK_SIZE' in data.columns
            and pd.notna(data['TICK_VALUE'].iloc[0]) and pd.notna(data['TICK_SIZE'].iloc[0])
        )
        carry_enabled = 'carry' in ModelsList and (carry_has_point_value or carry_has_tick) and (
            'far' in data.columns or ('investing_rate' in data.columns and 'funding_rate' in data.columns)
        )

        if carry_enabled:
            st.info('Running Carry Strategy')
            res = carry.calc(Inst_name, data, exchange_rate, point_value)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast
        else:
            if 'carry' in ModelsList and ('far' in data.columns or ('investing_rate' in data.columns and 'funding_rate' in data.columns)):
                st.warning('Skipping Carry Strategy because tick/point value information is missing.')

        NModels = len(StrategyName)
        if NModels == 0:
            st.warning(f"No strategies generated forecasts for {Inst_name}.")
            continue

        CorrMat = pd.DataFrame(CumList).T.corr()

        # Count models by strategy family
        ewma_count = sum(1 for key in AvgCapForecastDict if key.startswith("EWMA") and not key.startswith("EWMA_NORM"))
        carry_count = sum(1 for key in AvgCapForecastDict if key.startswith("CARRY"))
        ewma_norm_count = sum(1 for key in AvgCapForecastDict if key.startswith("EWMA_NORM"))

        st.write(f"EWMA models count: {ewma_count}")
        st.write(f"CARRY models count: {carry_count}")
        st.write(f"EWMA_NORM models count: {ewma_norm_count}")

        active_family_count = sum(count > 0 for count in [ewma_count, carry_count, ewma_norm_count])
        default_family_weight = 1.0 / active_family_count if active_family_count else 0.0

        ewma_weight = st.number_input(
            "EWMA Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_family_weight if ewma_count > 0 else 0.0,
            step=0.01,
            key=f"ewma_weight_{ins_name}",
            disabled=ewma_count == 0,
        )
        carry_weight = st.number_input(
            "CARRY Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_family_weight if carry_count > 0 else 0.0,
            step=0.01,
            key=f"carry_weight_{ins_name}",
            disabled=carry_count == 0,
        )
        ewma_norm_weight = st.number_input(
            "EWMA Norm Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_family_weight if ewma_norm_count > 0 else 0.0,
            step=0.01,
            key=f"ewma_norm_weight_{ins_name}",
            disabled=ewma_norm_count == 0,
        )
        biased_weights = {'EWMA': ewma_weight, 'CARRY': carry_weight, 'EWMA_NORM': ewma_norm_weight}

        # calculate biased weights for each strategy
        Weights = np.zeros(len(StrategyName))
        for i, name in enumerate(StrategyName):
            if name.startswith("EWMA_NORM") and ewma_norm_count > 0:
                Weights[i] = biased_weights['EWMA_NORM'] / ewma_norm_count
            elif name.startswith("EWMA") and ewma_count > 0:
                Weights[i] = biased_weights['EWMA'] / ewma_count 
            elif name.startswith("CARRY") and carry_count > 0:
                Weights[i] = biased_weights['CARRY'] / carry_count
            else:
                Weights[i] = 1.0 / NModels

        if sum(Weights) > 1.0 or sum(Weights) < .99:
            st.error("Weights sum to more than or less than 1.0, please check your weights calculation.")
            continue

        st.write(f"Controlled Weights: {Weights}")
        multiplier = min(1.0 / np.sqrt(np.dot(Weights.T, np.dot(CorrMat, Weights))), 2.5)
        UnweightedForecast = np.dot(Weights, AvgCapForecastList)
        FinalForecast = multiplier * UnweightedForecast

        st.write("### Forecast Results")
        st.write(f"Strategies: {StrategyName}")
        st.write(f"Strategy Weights: {Weights}")
        st.write(f"Multiplier: {multiplier}")
        st.write(f"Unweighted Forecast: {UnweightedForecast}")
        st.write(f"Weighted Forecast: {FinalForecast}")
        st.write("Correlation Matrix:")
        st.dataframe(CorrMat)

        progress_bar.progress(inst_idx / total_instruments if total_instruments else 1.0)
        progress_status.info(_progress_text("Main analysis instruments", inst_idx, total_instruments, analysis_start))

    progress_bar.progress(1.0)
    progress_status.success(_progress_text("Main analysis instruments", total_instruments, total_instruments, analysis_start))
