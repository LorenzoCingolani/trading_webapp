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
        
               
    # dictionary for storing models 
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
        AvgCapForecastDict = {}
        if 'ewma01' in ModelsList and 'ewma03' in ModelsList:
            
            print('Running EWMA Strategy')
            ResList = ewma.calc(Inst_name, data, MAParam, Standard_Cost, exchange_rate, point_value)
            passed_ewma_strategies = ResList.pop() # added to get the passed strategies

            save.h5file(resfold, savecode, *ResList)
            print(f"Passed EWMA strategies: {passed_ewma_strategies}")

            for param in MAParam:
                for res in ResList:
                    if res.name == f"EWMA{param:03d}":
                        if res.name in passed_ewma_strategies:
                            StrategyName.append(res.name)
                            CumList.append(res.cum_series)
                            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
                            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast
                        
            

 
        if 'breakout' in ModelsList and False:
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
                        AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        if 'carry' in ModelsList and 'far' in data.columns:
            print('Running Carry Strategy')
            res = carry.calc(Inst_name, data, exchange_rate, point_value)
            save.h5file(resfold, savecode, res)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        if 'stoch_in' in ModelsList and False:
            print('Not Running Stochastic Strategy')
            res = stoch.calc(Inst_name, data)
            save.h5file(resfold, savecode, res)
            StrategyName.append(res.name)
            CumList.append(res.cum_series)
            AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
            AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        if 'break01' in ModelsList and 'break03' in ModelsList and False:
            LookBackList = [5, 10, 15, 20]  # example lookbacks
            print('Running Break Model Strategy')
            ResList = break_model.calc(Inst_name, data, LookBackList)
            save.h5file(resfold, savecode, *ResList)

            for res in ResList:
                StrategyName.append(res.name)
                CumList.append(res.cum_series)
                AvgCapForecastList.append(res.avg_abs_val_capped_forecast)
                AvgCapForecastDict[res.name] = res.avg_abs_val_capped_forecast

        NModels = len(passed_ewma_strategies) + 1 
        
        if NModels == 0:
            print(f"No strategies generated forecasts for {Inst_name}.")
            continue

        CorrMat = pd.DataFrame(CumList).T.corr()
<<<<<<< HEAD
        Weights = np.ones(NModels) / NModels
        #import pdb; pdb.set_trace()
=======


        # Count models starting with "EWMA", "CARRY", and "Stochastic"
        ewma_count = len(passed_ewma_strategies)

        carry_count = sum(1 for key in AvgCapForecastDict if key.startswith("CARRY"))
        stochastic_count = sum(1 for key in AvgCapForecastDict if key.startswith("Stochastic"))

        # Print the counts
        print(f"EWMA models count: {ewma_count}")
        print(f"CARRY models count: {carry_count}")

        # calculate biased weights for each strategy

        biased_weights = {'EWMA': 0.5, 'CARRY': 0.5}
        


        # Weights = np.ones(NModels) / NModels
        # above is the default, but you can change it based on your strategy

        
        Weights = np.zeros(len(StrategyName))
        for i, name in enumerate(StrategyName):
            if name.startswith("EWMA"):
                Weights[i] = biased_weights['EWMA'] / ewma_count 
            elif name.startswith("CARRY"):
                Weights[i] = biased_weights['CARRY'] / carry_count
            else:
                Weights[i] = 1.0 / NModels
        if sum(Weights) > 1.0 or sum(Weights) < .99:
            raise ValueError("Weights sum to more than or less than 1.0, please check your weights calculation.")
        


        print(f"Controlled Weights: {Weights}")
>>>>>>> 1c789712e2bafd18bc1bdd56eb907a9d9f818b90
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
