import math
import os

import numpy as np
import pandas as pd
from django.conf import settings

def get_col_data(
    data: pd.DataFrame,
    col_name: str,
    date_col: str = 'Date',
    date_format: str = '%Y-%m-%d'
) -> pd.Series:
    """
    Extracts a column from a DataFrame, setting the date column as index.
    """
    if date_col not in data.columns:
        data = data.reset_index().copy()
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in the DataFrame.")
    data.dropna(subset=[date_col], inplace=True)
    data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    data.set_index(date_col, inplace=True)
    return data[col_name]

def compute_pnls(
    trades_needed: pd.Series,
    current_pos: pd.Series,
    fm: pd.DataFrame,
    px_closes: pd.Series,
    px_closes_prev: pd.Series
) -> tuple[list[float], list[float]]:
    """
    Computes PnLs for both new trades and current positions.
    """
    daily_instrument_pnls = []
    for ind, trade in trades_needed.items():
        if pd.isna(trade):
            daily_instrument_pnls.append(np.nan)
        else:
            tick_value = fm.loc[ind]['TICK_VALUE']
            tick_size = fm.loc[ind]['TICK_SIZE']
            fill_price = px_closes[ind] * (1 + 0.01 * -np.sign(trade))
            pnl = (px_closes[ind] - fill_price) * tick_value / tick_size * trade
            daily_instrument_pnls.append(pnl)

    daily_current_pnls = []
    for ind, cur_pos in current_pos.items():
        if not np.isnan(px_closes_prev[ind]):
            tick_value = fm.loc[ind]['TICK_VALUE']
            tick_size = fm.loc[ind]['TICK_SIZE']
            pnl = (px_closes[ind] - px_closes_prev[ind]) * tick_value / tick_size * cur_pos
            daily_current_pnls.append(pnl)
        else:
            daily_current_pnls.append(0)

    return daily_instrument_pnls, daily_current_pnls

def compute_trades(
    px_closes: pd.Series,
    px_closes_prev: pd.Series,
    std_dev: pd.Series,
    alpha_forecast: pd.Series,
    PDM: pd.Series,
    alpha_current_pos: pd.Series,
    fm: pd.DataFrame,
    cash_vol_daily: list
) -> tuple:
    """
    Computes position sizes, trades, and associated PnLs.
    """
    one_perc_change = px_closes * 0.01
    block_value = one_perc_change * fm['POINT_VALUE']
    price_volatility = np.round((std_dev / px_closes) * 100, 2)

    icv = price_volatility * block_value
    ivv = icv * fm['EXCHANGE_RATE']
    vol_scalar = cash_vol_daily / ivv
    pos_contracts = vol_scalar * alpha_forecast / 10

    target_pos = np.round(pos_contracts * PDM * fm['INSTRUMENT_WEIGHTS'])
    trades_needed = target_pos - alpha_current_pos
    daily_instrument_pnls, daily_current_pnls = compute_pnls(trades_needed, alpha_current_pos, fm, px_closes, px_closes_prev)

    return (
        one_perc_change, block_value, price_volatility, cash_vol_daily,
        std_dev, icv, ivv, vol_scalar,
        pos_contracts, trades_needed, target_pos,
        daily_instrument_pnls, daily_current_pnls
    )

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
    """
    Main framework function to compute trades based on forecasted data.
    """
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
    fm = fm[fm['INSTRUMENT'].isin(all_alpha_forecast.keys())]

    trades = []
    current_pos = pd.Series([0] * fm.shape[0], index=fm.index)
    alpha_current_pos = pd.Series([0] * fm.shape[0], index=fm.index)
    px_closes_prev = pd.Series([np.nan] * fm.shape[0], index=fm.index)

    for date in alpha_forecast_df.index:
        alpha_forecast = alpha_forecast_df.loc[date].astype(float)
        px_closes = px_close_df.loc[date].astype(float)
        std_dev = std_dev_df.loc[date].astype(float)
        cash_vol_tgt_daily = [aum * 0.2 / math.sqrt(256)] * len(alpha_forecast)

        values = list(compute_trades(px_closes, px_closes_prev, std_dev, alpha_forecast,
                                     PDM, alpha_current_pos, fm, cash_vol_tgt_daily))
        values.insert(0, alpha_forecast)

        px_closes_prev = px_closes
        output = []

        ret = np.array([list(val) for val in values])
        for j in range(ret.shape[1]):
            for i in range(ret.shape[0]):
                output.append(ret[i][j])

        trades.append(output)

        alpha_current_pos = pd.Series(np.nan_to_num(values[-6]), index=fm.index)
        current_pos = pd.Series(np.nan_to_num(values[-3]), index=fm.index)
        daily_instrument_pnls = values[-2]
        aum += np.nansum(daily_instrument_pnls)
        aums.append(aum)

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
