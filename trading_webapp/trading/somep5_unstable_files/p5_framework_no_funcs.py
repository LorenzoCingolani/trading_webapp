import math
import os

import numpy as np
import pandas as pd
from django.conf import settings

def get_col_data(data: pd.DataFrame, col_name: str, date_col: str = 'Date', date_format: str = '%Y-%m-%d') -> pd.Series:
    if date_col not in data.columns:
        data = data.reset_index().copy()
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in the DataFrame.")
    data.dropna(subset=[date_col], inplace=True)
    data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    data.set_index(date_col, inplace=True)
    return data[col_name]

def framework_main(
    fm: dict,
    combinedForcastFolderPath: str,
    csv_dictionary: dict,
    PDM: pd.Series,
    date_format: str,
    aum: float,
    is_markov: bool = False,
    std_dev_days: int = 20
) -> pd.DataFrame:
    aums = []
    all_alpha_forecast, all_px_closes, all_std_dev = {}, {}, {}
    product_list = list(csv_dictionary.keys())

    for instrument in product_list:
        print(f"Processing {instrument}...")
        try:
            source_data = csv_dictionary[instrument]
            if is_markov:
                all_alpha_forecast[instrument] = get_col_data(source_data, 'MarkovAdjFinalForecast')
                all_std_dev[instrument] = get_col_data(source_data, 'PX_CLOSE_1D', 'Date', date_format).rolling(std_dev_days).std()
            else:
                forecast_path = os.path.join(combinedForcastFolderPath, f"{instrument}.csv")
                forecast_data = pd.read_csv(forecast_path)
                all_alpha_forecast[instrument] = get_col_data(forecast_data, 'FinalForecast')

            all_px_closes[instrument] = get_col_data(source_data, 'PX_CLOSE_1D', 'Date', date_format)
            all_std_dev[instrument] = get_col_data(source_data, 'st_dev', 'Date', date_format)
        except Exception as ex:
            print('Complete data is not available')
            print(f'Caught {ex}')

    alpha_forecast_df = pd.concat(all_alpha_forecast.values(), axis=1, keys=all_alpha_forecast.keys())
    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    std_dev_df = pd.concat(all_std_dev.values(), axis=1, keys=all_std_dev.keys())

    if not isinstance(fm, pd.DataFrame):
        fm = pd.DataFrame(fm).T
    fm = fm[fm['INSTRUMENT'].isin(alpha_forecast_df.columns)]

    trades = []
    current_pos = pd.Series(0, index=fm['INSTRUMENT'])
    alpha_current_pos = pd.Series(0, index=fm['INSTRUMENT'])
    px_closes_prev = pd.Series(np.nan, index=fm['INSTRUMENT'])

    for date in alpha_forecast_df.index:
        alpha_forecast = alpha_forecast_df.loc[date]
        px_closes = px_close_df.loc[date]
        std_dev = std_dev_df.loc[date]

        one_perc_change = px_closes * 0.01
        block_value = one_perc_change * fm.set_index('INSTRUMENT')['POINT_VALUE']
        price_volatility = (std_dev / px_closes * 100).round(2)

        icv = price_volatility * block_value
        ivv = icv * fm.set_index('INSTRUMENT')['EXCHANGE_RATE']
        cash_vol_daily = aum * 0.2 / math.sqrt(256)
        vol_scalar = cash_vol_daily / ivv
        pos_contracts = vol_scalar * alpha_forecast / 10

        target_pos = (pos_contracts * PDM * fm.set_index('INSTRUMENT')['INSTRUMENT_WEIGHTS']).round()
        trades_needed = target_pos - alpha_current_pos

        fill_price = px_closes.copy()
        fill_price[trades_needed > 0] *= 1.01
        fill_price[trades_needed < 0] *= 0.99

        tick_value = fm.set_index('INSTRUMENT')['TICK_VALUE']
        tick_size = fm.set_index('INSTRUMENT')['TICK_SIZE']

        daily_instrument_pnls = ((px_closes - fill_price) * (tick_value / tick_size) * trades_needed).fillna(0)
        daily_current_pnls = ((px_closes - px_closes_prev) * (tick_value / tick_size) * current_pos).fillna(0)

        values = [
            alpha_forecast, one_perc_change, block_value, price_volatility,
            pd.Series(cash_vol_daily, index=alpha_forecast.index),
            std_dev, icv, ivv, vol_scalar,
            pos_contracts, trades_needed, target_pos,
            daily_instrument_pnls, daily_current_pnls
        ]

        px_closes_prev = px_closes
        alpha_current_pos = target_pos
        current_pos = target_pos
        aum += daily_instrument_pnls.sum()
        aums.append(aum)

        ret = np.array([v.values for v in values])
        trades.append(ret.T.flatten().tolist())

    out = [
        'markov_forecast' if is_markov else 'alpha_forecast',
        'one_perc_change', 'block_value', 'price_volatility',
        'cash_vol_daily', 'std_dev', 'icv', 'ivv', 'vol_scalar',
        'markov_subsystem_position' if is_markov else 'alpha_subsystem_position',
        'markov_trades_needed' if is_markov else 'alpha_trades_needed',
        'markov_target_pos' if is_markov else 'alpha_target_pos',
        'daily_instrument_pnls', 'daily_current_pnls'
    ]

    val_cols = alpha_forecast_df.columns.tolist()
    new_cols = [f'{col}_{feature}' for col in val_cols for feature in out]

    trades_df = pd.DataFrame.from_records(
        trades,
        index=alpha_forecast_df.index,
        columns=new_cols
    )
    trades_df['AUM'] = aums
    return trades_df
