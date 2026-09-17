import math
import os
import numpy as np
import pandas as pd
import streamlit as st

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
    def get_col_data(data: pd.DataFrame, col_name: str, date_col: str = 'Date', date_format: str = '%Y-%m-%d') -> pd.Series:
        if date_col not in data.columns:
            data = data.reset_index().copy()
        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in the DataFrame.")
        data.dropna(subset=[date_col], inplace=True)
        data[date_col] = pd.to_datetime(data[date_col], format=date_format)
        data.set_index(date_col, inplace=True)
        return data[col_name]

    aums = []
    navs = []
    all_alpha_forecast, all_px_closes, all_std_dev = {}, {}, {}
    product_list = list(csv_dictionary.keys())

    st.info("Processing instruments for forecast generation...")
    progress_bar = st.progress(0)
    for idx, instrument in enumerate(product_list):
        st.write(f"Processing {instrument}...")
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
            st.warning(f'Complete data is not available for {instrument}: {ex}')
        progress_bar.progress((idx + 1) / len(product_list))

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

    st.info("Running forecast calculations...")
    date_list = list(alpha_forecast_df.index)
    progress_bar2 = st.progress(0)

    for idx, date in enumerate(date_list):
        alpha_forecast = alpha_forecast_df.loc[date].astype(float)
        px_closes = px_close_df.loc[date].astype(float)
        std_dev = std_dev_df.loc[date].astype(float)

        one_perc_change = px_closes * 0.01
        block_value = one_perc_change * fm['POINT_VALUE']
        price_volatility = np.round((std_dev / px_closes) * 100, 2)
        icv = price_volatility * block_value
        ivv = icv * fm['EXCHANGE_RATE']

        # Same aum snapshot (as of the start of this day, before today's P&L) feeds both -
        # vol_scalar is what actually sizes today's position, and cash_vol_tgt_daily is just
        # that same calc's numerator, shown for reference. They must match within a row:
        # cash_vol_tgt_daily / ivv == vol_scalar. AUM still updates with P&L below exactly as
        # before - this only fixes which row displays which AUM snapshot, not whether AUM
        # compounds day to day.
        cash_vol_tgt_daily = [aum * 0.2 / math.sqrt(256)] * len(alpha_forecast)
        vol_scalar = (aum * 0.2 / math.sqrt(256)) / ivv
        pos_contracts = vol_scalar * alpha_forecast / 10
        target_pos = pos_contracts * PDM * fm['INSTRUMENT_WEIGHTS']
        target_pos = pd.to_numeric(target_pos, errors='coerce').fillna(0).round().astype(int)
        trades_needed = target_pos - alpha_current_pos
        trades_needed = pd.to_numeric(trades_needed, errors='coerce').round().fillna(0).astype(int)

        execution_price = pd.Series(
            np.where(
                trades_needed < 0, px_closes * 0.99,
                np.where(
                    trades_needed > 0, px_closes * 1.01,
                    np.where(trades_needed == 0, px_closes, 0)
                )
            ),
            index=px_closes.index
        )

        daily_instrument_pnls = []
        for ind, trade in trades_needed.items():
            if pd.isna(trade):
                daily_instrument_pnls.append(np.nan)
            else:
                tick_value = fm.loc[ind]['TICK_VALUE']
                tick_size = fm.loc[ind]['TICK_SIZE']
                fill_price = px_closes[ind] * (1 + (0.01 * np.sign(trade)))
                pnl_1 = (px_closes[ind] - fill_price) * tick_value / tick_size * trade
                daily_instrument_pnls.append(pnl_1)

        daily_current_pnls = []
        for ind, cur_pos in current_pos.items():
            if not np.isnan(px_closes_prev[ind]):
                tick_value = fm.loc[ind]['TICK_VALUE']
                tick_size = fm.loc[ind]['TICK_SIZE']
                pnl_2 = (px_closes[ind] - px_closes_prev[ind]) * tick_value / tick_size * cur_pos
                daily_current_pnls.append(pnl_2)
            else:
                daily_current_pnls.append(0)

        # AUM updates with today's P&L here, after today's sizing decision is already locked
        # in above - this is what makes tomorrow's vol_scalar/cash_vol_tgt_daily reflect
        # today's result, i.e. AUM still compounds day to day exactly as before.
        #
        # daily_instrument_pnls (the execution/slippage leg) is deliberately NOT included here:
        # execution_price currently assumes a flat 1% price impact on every trade, which isn't
        # a real fill and isn't truthful P&L - it would just distort AUM/Sharpe with an
        # arbitrary number. Real trading costs are already captured properly elsewhere (Cost %
        # on the Sharpe Ratio page, using each instrument's actual STANDARD_COST). Once
        # execution_price is replaced with a real broker fill, add
        # np.nansum(daily_instrument_pnls) back in here - the logic/column stays, just not
        # feeding AUM for now.
        aum += np.nansum(daily_current_pnls)
        aums.append(aum)
        nav = aum.copy()
        navs.append(nav)

        px_closes_prev = px_closes

        values = [
            alpha_forecast, one_perc_change, block_value, price_volatility,
            cash_vol_tgt_daily, std_dev, icv, ivv, vol_scalar,
            pos_contracts, trades_needed, target_pos,
            execution_price, daily_instrument_pnls, daily_current_pnls,
            alpha_current_pos
        ]

        output = []
        ret = np.array([list(val) for val in values])
        for j in range(ret.shape[1]):
            for i in range(ret.shape[0]):
                output.append(ret[i][j])
        trades.append(output)

        alpha_current_pos = pd.Series(np.nan_to_num(target_pos), index=fm.index).fillna(0)
        current_pos = pd.Series(np.nan_to_num(target_pos), index=fm.index)

        progress_bar2.progress((idx + 1) / len(date_list))

    out = [
        'markov_forecast' if is_markov else 'alpha_forecast',
        'one_perc_change', 'block_value', 'price_volatility',
        'cash_vol_daily', 'std_dev', 'icv', 'ivv', 'vol_scalar',
        'markov_subsystem_position' if is_markov else 'alpha_subsystem_position',
        'markov_trades_needed' if is_markov else 'alpha_trades_needed',
        'markov_target_pos' if is_markov else 'alpha_target_pos',
        'execution_price',
        'pnl_intraday_trades', 'pnl_carried_forward',
        'alpha_current_pos'
    ]

    val_cols = alpha_forecast_df.columns.tolist()
    new_cols = [f'{col}_{feature}' for col in val_cols for feature in out]

    trades_df = pd.DataFrame.from_records(
        trades,
        index=alpha_forecast_df.index,
        columns=new_cols
    )
    # Portfolio-level breakdown, so you don't have to manually sum the per-instrument pnl
    # columns yourself:
    # - total_pnl_intraday_trades: cost of each instrument's rebalancing trade that day -
    #   reference only, NOT included in AUM (see the note above execution_price's 1% guess)
    # - total_pnl_carried_forward: mark-to-market on positions already held overnight - this
    #   is what actually drives AUM right now
    # - total_pnl_today: the AUM-driving total, i.e. == total_pnl_carried_forward for now.
    #   Once daily_instrument_pnls uses a real execution price, this becomes the sum of both.
    instrument_trade_pnl_cols = [c for c in new_cols if c.endswith('_pnl_intraday_trades')]
    carried_forward_pnl_cols = [c for c in new_cols if c.endswith('_pnl_carried_forward')]
    trades_df['total_pnl_intraday_trades'] = trades_df[instrument_trade_pnl_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    trades_df['total_pnl_carried_forward'] = trades_df[carried_forward_pnl_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    trades_df['total_pnl_today'] = trades_df['total_pnl_carried_forward']

    trades_df['AUM'] = aums
    trades_df['NAV'] = navs

    st.success("Forecast calculations complete!")
    st.write("Preview of generated trades/orders:")
    st.dataframe(trades_df.head(20))

    return trades_df
