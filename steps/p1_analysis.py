import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict

from strategies import ewma, break_model, carry, stoch
from strategies import stochastic_breakout as breakout

def main_analysis(framework_dict: Dict[str, Dict[str, float]], 
                  csvs_dictionary: Dict[str, pd.DataFrame]) -> None:
    """
    Run financial strategy models (EWMA, Breakout, Carry, etc.) on instruments' data.
    Display results using Streamlit.
    """
    ModelsList = ['ewma01', 'ewma02', 'ewma03', 'ewma04', 'breakout', 'carry']
    MAParam = [2, 4, 8, 16]
    BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]

    for ins_name, data in csvs_dictionary.items():
        st.subheader(f'Instrument name: {ins_name}')

        if ins_name not in framework_dict:
            st.warning(f"No parameters found for {ins_name}")
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

        if 'ewma01' in ModelsList and 'ewma03' in ModelsList:
            st.info('Running EWMA Strategy')
            ResList = ewma.calc(Inst_name, data, MAParam, Standard_Cost, exchange_rate, point_value)
            passed_ewma_strategies = ResList.pop()  # last item is the list of passed strategies

            st.write(f"Passed EWMA strategies: {passed_ewma_strategies}")

            for param in MAParam:
                for res in ResList:
                    if res.name == f"EWMA{param:03d}":
                        if res.name in passed_ewma_strategies:
                            StrategyName.append(res.name)
                            CumList.append(res.cum_series)
                            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
                            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        if 'carry' in ModelsList and 'far' in data.columns:
            st.info('Running Carry Strategy')
            res = carry.calc(Inst_name, data, exchange_rate, point_value)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

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

        # calculate biased weights for each strategy
        biased_weights = {'EWMA': 0.8, 'CARRY': 0.2}

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