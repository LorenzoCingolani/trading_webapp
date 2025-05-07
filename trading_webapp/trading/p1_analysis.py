import os
import numpy as np
import pandas as pd
from typing import Dict
from django.conf import settings

from .strategies import ewma, break_model, carry, save, stoch
from .strategies import stochastic_breakout as breakout


def main_analysis(framework_dict: Dict[str, Dict[str, float]], 
                  csvs_dictionary: Dict[str, pd.DataFrame], 
                  MainFolderPath: str) -> None:
    """
    Run financial strategy models (EWMA, Breakout, Carry, etc.) on instruments' data.

    Args:
        framework_dict (Dict[str, Dict[str, float]]): Dictionary mapping instrument names to parameter dictionaries.
        csvs_dictionary (Dict[str, pd.DataFrame]): Dictionary mapping instrument names to their time series data.
        MainFolderPath (str): Path to the main output folder.

    Returns:
        None
    """

    ModelsList = ['ewma01', 'ewma02', 'ewma03', 'ewma04', 'breakout', 'carry']
    MAParam = [2, 4, 8, 16]
    BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]
    resfold = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
    # delete all old from resfold
    for file in os.listdir(resfold):
        os.remove(os.path.join(resfold, file))
        
               

    for ins_name, data in csvs_dictionary.items():
        print(f'Instrument name: {ins_name}')
        

        if ins_name not in framework_dict:
            print(f"No parameters found for {ins_name}")
            continue

        params = framework_dict[ins_name]
        Inst_name = params['INSTRUMENT']
        Standard_Cost = params['STANDARD_COST']
        exchange_rate = params['EXCHANGE_RATE']
        point_value = params['POINT_VALUE']

        print(f"--- Analyzing {Inst_name} ---")

        if Inst_name not in csvs_dictionary:
            print(f'No data found for {Inst_name}. Available keys: {list(csvs_dictionary.keys())}')
            continue

        data = csvs_dictionary[Inst_name].copy()
        savecode = f"{Inst_name}_out.h5"

        Params = save.Output('params')
        Params.models = ModelsList
        save.h5file(resfold, savecode, Params)

        StrategyName = []
        CumList = []
        AvgCapForecastList = []
        if 'ewma01' in ModelsList and 'ewma03' in ModelsList:
            
            print('Running EWMA Strategy')
            ResList = ewma.calc(Inst_name, data, MAParam, Standard_Cost, exchange_rate, point_value)
            
            save.h5file(resfold, savecode, *ResList)

            for param in MAParam:
                for res in ResList:
                    if res.name == f"EWMA{param:03d}":
                        StrategyName.append(res.name)
                        CumList.append(res.cum_series)
                        AvgCapForecastList.append(res.avg_abs_val_capped_forecast)

        if 'breakout' in ModelsList:
            print('Running Breakout Strategy')
            ResList = breakout.calc(Inst_name, data, BreakParam, Standard_Cost, exchange_rate, point_value)
            save.h5file(resfold, savecode, *ResList)
            Params.BreakoutBest = BreakParam
            save.h5file(resfold, savecode, Params)

            for param in BreakParam:
                for res in ResList:
                    if res.name == f"Stochastic Breakout{param[0]:.3f}_{param[1]:03d}":
                        StrategyName.append(res.name)
                        CumList.append(res.cum_series)
                        AvgCapForecastList.append(res.avg_abs_val_capped_forecast)

        if 'carry' in ModelsList and 'far' in data.columns:
            print('Running Carry Strategy')
            res = carry.calc(Inst_name, data, exchange_rate, point_value)
            save.h5file(resfold, savecode, res)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)

        if 'stoch_in' in ModelsList:
            print('Running Stochastic Strategy')
            res = stoch.calc(Inst_name, data)
            save.h5file(resfold, savecode, res)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)

        if 'break01' in ModelsList and 'break03' in ModelsList:
            LookBackList = [5, 10, 15, 20]  # example lookbacks
            print('Running Break Model Strategy')
            ResList = break_model.calc(Inst_name, data, LookBackList)
            save.h5file(resfold, savecode, *ResList)

            for res in ResList:
                StrategyName.append(res.name)
                CumList.append(res.cum_series)
                AvgCapForecastList.append(res.avg_abs_val_capped_forecast)

        NModels = len(AvgCapForecastList)
        if NModels == 0:
            print(f"No strategies generated forecasts for {Inst_name}.")
            continue

        CorrMat = pd.DataFrame(CumList).T.corr()
        Weights = np.ones(NModels) / NModels
        multiplier = min(1.0 / np.sqrt(np.dot(Weights.T, np.dot(CorrMat, Weights))), 2.5)
        UnweightedForecast = np.dot(Weights, AvgCapForecastList)
        FinalForecast = multiplier * UnweightedForecast

        Forecast = save.Output('forecast')
        Forecast.strategies = StrategyName
        Forecast.strategy_weight = Weights
        Forecast.multiplier = multiplier
        Forecast.unweighted_forecast = UnweightedForecast
        Forecast.weighted_forecast = FinalForecast
        Forecast.CorrelationMatrix = CorrMat

        save.h5file(resfold, savecode, Forecast)
