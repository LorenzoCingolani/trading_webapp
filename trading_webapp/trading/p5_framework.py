# ### Framework
# 
# Tells the trades needed.
# 
# Before running this notebook, run first
# - portfolio_diver_mult
# - combined_validation_forecast
# - portfolio_diver_multiplier
# - adjust_markov_movements

import math
import os

import numpy as np
import pandas as pd

from django.conf import settings


def get_col_data(data, col_name, date_col='Date',date_format='%Y-%m-%d'):
    if date_col not in data.columns:
        data = data.reset_index().copy()
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in the DataFrame.")
    data.dropna(subset=[date_col],inplace=True)
    data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    data.set_index(date_col, inplace=True)
    return data[col_name]


def compute_pnls(trades_needed, current_pos, fm, px_closes, px_closes_prev):
    daily_instrument_pnls = []
    for ind,trade in enumerate(trades_needed):
        if pd.isna(trade):
            daily_instrument_pnls.append(np.nan)
        else:
            if trade >0:
                buy_price = px_closes[ind] * 1.01 # Placeholder as while selling we will sell it above px_closes
                daily_instrument_pnl = (px_closes[ind] - buy_price)*fm.iloc[ind]['TIC_VALUE']/fm.iloc[ind]['TIC_SIZE']
            elif trade <0:
                sell_price = 0.99 * px_closes[ind] # Placeholder as while selling we will sell it below px_closes
                daily_instrument_pnl = (sell_price-px_closes[ind])*fm.iloc[ind]['TIC_VALUE']/fm.iloc[ind]['TIC_SIZE']
            else:
                # print ('no_trade')
                daily_instrument_pnl = 0
            daily_instrument_pnls.append(daily_instrument_pnl)
    # Current position pnl
    # =IF(AX22<0,((BA22-BA21)*10/0.1)*AX22,IF(AX22>0,(BA21-BA22)*10/0.1, 0))
    daily_current_pnls = []
    for ind, cur_pos in current_pos.items():
        if not np.isnan(px_closes_prev[ind]):
            if cur_pos < 0:
                daily_current_pnl = (px_closes[ind]- px_closes_prev[ind] ) * fm.iloc[ind]['TIC_VALUE']/fm.iloc[ind]['TIC_SIZE'] * abs(cur_pos)
            elif cur_pos > 0:
                daily_current_pnl = ( px_closes_prev[ind]- px_closes[ind]) * fm.iloc[ind]['TIC_VALUE']/fm.iloc[ind]['TIC_SIZE'] * abs(cur_pos)
            else:
                daily_current_pnl = 0
            daily_current_pnls.append(daily_current_pnl)
        else:
            daily_current_pnls.append(0)

    return daily_instrument_pnls, daily_current_pnls


def compute_trades(px_closes, px_closes_prev, std_dev, alpha_forecast, PDM, alpha_current_pos, fm, cash_vol_daily):
    
    one_perc_change = px_closes*0.01
    block_value = one_perc_change * fm['POINT_VALUE']
    price_volatility = np.round((std_dev / px_closes) * 100, 2)
    
    icv = price_volatility * block_value
    ivv = icv * fm['EXCHANGE_RATE']
    vol_scalar = cash_vol_daily / ivv
    pos_contracts = vol_scalar * alpha_forecast/10 #subsystem position

    target_pos = np.round(pos_contracts * PDM*fm['INSTRUMENT_WEIGHTS'])
    trades_needed = target_pos - alpha_current_pos
    daily_instrument_pnls,  daily_current_pnls = compute_pnls(trades_needed, target_pos, fm, px_closes, px_closes_prev)
    return  one_perc_change, block_value, price_volatility, cash_vol_daily,\
            std_dev, icv, ivv, vol_scalar,\
            pos_contracts, trades_needed, target_pos,\
            daily_instrument_pnls,  daily_current_pnls
            # markov_pos_contracts, markov_trades_needed, markov_target_pos, \
            


def framework_main(fm, combinedForcastFolderPath, csv_dictionary, PDM,date_format, aum, is_markov=False, std_dev_days=20):
    aums = []

    all_alpha_forecast = {}
    all_px_closes = {}
    all_std_dev = {}

    
    product_list = list(csv_dictionary.keys())
    for instrument in product_list:
        inst_path = os.path.join(combinedForcastFolderPath, f"{instrument}.csv")
        print(f"Processing {instrument}...")
        
        try:
            if is_markov:
                all_alpha_forecast[instrument] = get_col_data(csv_dictionary[instrument], 'MarkovAdjFinalForecast')
                all_std_dev[instrument] = all_px_closes[instrument].rolling(std_dev_days).std()
            else:
                data_combined_forcast = pd.read_csv(inst_path)
                all_alpha_forecast[instrument] = get_col_data(data_combined_forcast, 'FinalForecast')
            print('data_combined_forcast:', data_combined_forcast)
            
            all_px_closes[instrument] = get_col_data(csv_dictionary[instrument], 'PX_CLOSE_1D', 'Date', date_format=date_format)
            # Compute sta
            # all_std_dev[instrument] = all_px_closes[instrument].rolling(std_dev_days).std()
            
            all_std_dev[instrument] = get_col_data(csv_dictionary[instrument], 'st_dev', 'Date', date_format=date_format)
            #all_std_dev[instrument] = get_col_data(csv_dictionary[instrument], 'st_dev', 'Date', date_format="%d/%m/%Y")
        except Exception as ex:
            print('Complete data is not available ')
            print(f'Caught {ex}')
    
    alpha_forecast_df = pd.concat(all_alpha_forecast.values(), axis=1, keys=all_alpha_forecast.keys())
    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    
    print('see next variables',all_std_dev)
    std_dev_df = pd.concat(all_std_dev.values(), axis=1, keys=all_std_dev.keys())
    # std_dev_df = std_dev
    # Only working with framework values whose forecasts are available.
    
    fm = pd.DataFrame(fm).T # converting dictionary to dataframe
    fm = fm[fm['INSTRUMENT'].isin(all_alpha_forecast.keys())]
    trades = []
    
    current_pos = pd.Series([0]  * fm.shape[0])
    alpha_current_pos = 0
    px_closes_prev = pd.Series([np.nan] * fm.shape[0])
    px_closes_prev.index = fm['INSTRUMENT']
    for date in alpha_forecast_df.index:
        alpha_forecast = alpha_forecast_df.loc[date].values.astype(float)
        # markov_forecast = markov_forecast_df.loc[date].values.astype(float)
        px_closes = px_close_df.loc[date].values.astype(float)
        std_dev = std_dev_df.loc[date].values.astype(float)
        cash_vol_tgt_daily = [aum * 0.2/ math.sqrt(256)] * len(alpha_forecast)
        # std_dev = np.clip(std_dev, -3, 3) # clip the lowest value of standard deviation to 1%
        values = list(compute_trades(px_closes, px_closes_prev, std_dev, alpha_forecast, 
                                     PDM, current_pos, fm, cash_vol_tgt_daily))
        values.insert(0, alpha_forecast)
        
        px_closes_prev = px_closes
        output = []
        ret = np.array([list(val) for val in values])
        for j in range(ret.shape[1]):
            for i in range(ret.shape[0]):
                output.append(ret[i][j])

        trades.append(output)
        
        alpha_current_pos = values[-6].copy()
        alpha_current_pos = np.nan_to_num(alpha_current_pos)
        current_pos = values[-3].copy()
        current_pos = pd.Series(np.nan_to_num(current_pos))
        daily_instrument_pnls = values[-2]
        aum = aum + np.nansum(daily_instrument_pnls) 
        aums.append(aum)

    if is_markov:
        out = [
            'markov_forecast',
            'one_perc_change', ' block_value', ' price_volatility',
            'cash_vol_daily', 'std_dev','icv', 'ivv', 'vol_scalar',
            'markov_subsystem_position', 'markov_trades_needed', 'markov_target_pos',
            'daily_instrument_pnls', 'daily_current_pnls'
        ]
    else:
        out = [
            'alpha_forecast', 
            'one_perc_change', ' block_value', ' price_volatility',
            'cash_vol_daily', 'std_dev','icv', 'ivv', 'vol_scalar',
            'alpha_subsystem_position', 'alpha_trades_needed', 'alpha_target_pos',
            'daily_instrument_pnls', 'daily_current_pnls'
        ]

    val_cols = alpha_forecast_df.columns.tolist() 
    new_cols = [f'{col}_{feature}' for col in val_cols for feature in out]

    trades = pd.DataFrame.from_records(
        trades,
        index=alpha_forecast_df.index, 
        columns=new_cols)
    trades['AUM'] = aums

    return trades

if __name__ == '__main__':
    
    # Read products and weights from framewrok input file
    framework_input_file=r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\trading_webapp\DATA\input_main\input_main_framework.csv'
    read_fold=r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\trading_webapp\DATA\combinedForecast'
    date_format="%d/%m/%Y"
    fm = pd.read_csv(framework_input_file)

    ### Read Portfolio Diversification Multiplier
    PDM=1.86 # copy from 3-PDM_portfolio.h5 file
    aum = 10_000_000

    fm=pd.read_csv(framework_input_file)
    print(fm.head())


    analysis_input_folder = r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\trading_webapp\DATA\input_instruments'
    
    csvs_dictionary = {}

    for file in os.listdir(analysis_input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(analysis_input_folder, file)
            df = pd.read_csv(file_path)
            html_table = df.to_html(classes='table table-bordered', index=False)
            # reove csv from end
            file = file[:-4]
            csvs_dictionary[file] = df.copy()


    results = framework_main(fm, read_fold, csvs_dictionary,  PDM, date_format, aum, is_markov=False)
    print('Running for markov movements')
    print('results:', results)
    #results.to_csv(r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\trading_webapp\DATA\order_folder\trade_alpha.csv')
    #main(fm, read_fold, input_folder,  PDM, date_format, aum, is_markov=True)

    