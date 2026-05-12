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

    MAParam = [2, 4, 8, 16]
    BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]

    EwmaResult = namedtuple("EwmaResult", ["name", "cum_series", "avg_abs_val_capped_forecast"])

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
                res = EwmaResult(
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

        # Count models starting with "EWMA" and "CARRY"
        ewma_count = sum(1 for key in AvgCapForecastDict if key.startswith("EWMA"))
        carry_count = sum(1 for key in AvgCapForecastDict if key.startswith("CARRY"))

        st.write(f"EWMA models count: {ewma_count}")
        st.write(f"CARRY models count: {carry_count}")

        if ewma_count > 0 and carry_count == 0:
            default_ewma_weight, default_carry_weight = 1.0, 0.0
        elif carry_count > 0 and ewma_count == 0:
            default_ewma_weight, default_carry_weight = 0.0, 1.0
        else:
            default_ewma_weight, default_carry_weight = 0.5, 0.5

        ewma_weight = st.number_input(
            "EWMA Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_ewma_weight,
            step=0.01,
            key=f"ewma_weight_{ins_name}",
            disabled=ewma_count == 0,
        )
        carry_weight = st.number_input(
            "CARRY Weight",
            min_value=0.0,
            max_value=1.0,
            value=default_carry_weight,
            step=0.01,
            key=f"carry_weight_{ins_name}",
            disabled=carry_count == 0,
        )
        biased_weights = {'EWMA': ewma_weight, 'CARRY': carry_weight}

        # calculate biased weights for each strategy
        Weights = np.zeros(len(StrategyName))
        for i, name in enumerate(StrategyName):
            if name.startswith("EWMA") and ewma_count > 0:
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
