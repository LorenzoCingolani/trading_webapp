import math
import os
import numpy as np
import pandas as pd
import streamlit as st

from steps.volatility import simple_price_volatility
from steps.fx_utils import to_usd, DIRECT_QUOTE


def _first_number(data: pd.DataFrame, col: str) -> float:
    if col not in data.columns:
        return np.nan
    values = pd.to_numeric(data[col], errors='coerce').dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def trade_costs(data: pd.DataFrame):
    """(half the bid-ask spread in price points, execution + clearing per contract in the
    instrument's own currency), from the input file's BID/ASK/EXECUTION_COST/CLEARING_COST.
    Same inputs as the Standard Cost. Missing spread -> NaN (caller falls back to half a tick)."""
    half_spread = (_first_number(data, 'ASK') - _first_number(data, 'BID')) / 2
    commission = np.nansum([_first_number(data, 'EXECUTION_COST'), _first_number(data, 'CLEARING_COST')])
    return half_spread, float(commission)


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
    costs_by_instrument = {}
    product_list = list(csv_dictionary.keys())

    st.info("Processing instruments for forecast generation...")
    progress_bar = st.progress(0)
    for idx, instrument in enumerate(product_list):
        st.write(f"Processing {instrument}...")
        try:
            source_data = csv_dictionary[instrument]
            costs_by_instrument[instrument] = trade_costs(source_data)
            if is_markov:
                all_alpha_forecast[instrument] = get_col_data(source_data, 'MarkovAdjFinalForecast')
                all_std_dev[instrument] = get_col_data(source_data, 'PX_CLOSE_1D', 'Date', date_format).rolling(std_dev_days).std()
            else:
                forecast_path = os.path.join(combinedForcastFolderPath, f"{instrument}.csv")
                forecast_data = pd.read_csv(forecast_path)
                all_alpha_forecast[instrument] = get_col_data(forecast_data, 'FinalForecast')
            all_px_closes[instrument] = get_col_data(source_data, 'PX_CLOSE_1D', 'Date', date_format)
            all_std_dev[instrument] = simple_price_volatility(all_px_closes[instrument])
        except Exception as ex:
            st.warning(f'Complete data is not available for {instrument}: {ex}')
        progress_bar.progress((idx + 1) / len(product_list))

    alpha_forecast_df = pd.concat(all_alpha_forecast.values(), axis=1, keys=all_alpha_forecast.keys())
    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    std_dev_df = pd.concat(all_std_dev.values(), axis=1, keys=all_std_dev.keys())

    # A day with no price for an instrument (e.g. a US holiday when Eurex is open): keep its
    # position and last close, instead of closing to 0 and reopening the next day (which lost
    # the move across the holiday and paid a round trip for nothing).
    has_price = px_close_df.notna()
    px_close_df = px_close_df.ffill()
    std_dev_df = std_dev_df.ffill()
    alpha_forecast_df = alpha_forecast_df.ffill()

    if not isinstance(fm, pd.DataFrame):
        fm = pd.DataFrame(fm).T
    fm = fm[fm['INSTRUMENT'].isin(all_alpha_forecast.keys())]

    # Trading cost per contract traded: fill at the close +/- half the spread, plus execution +
    # clearing. Replaces the old flat 1% placeholder fill.
    tick_sizes = pd.to_numeric(fm['TICK_SIZE'], errors='coerce')
    half_spread = pd.Series({i: costs_by_instrument.get(i, (np.nan, 0.0))[0] for i in fm.index}, dtype=float)
    no_spread = half_spread.index[half_spread.isna() | (half_spread <= 0)].tolist()
    if no_spread:
        st.warning("No BID/ASK in the input file for " + ", ".join(no_spread) +
                   " - assuming a 1-tick spread for trading costs.")
        half_spread[no_spread] = tick_sizes[no_spread] / 2
    commission = pd.Series({i: costs_by_instrument.get(i, (np.nan, 0.0))[1] for i in fm.index}, dtype=float)

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
        quote_convention = fm['QUOTE_CONVENTION'] if 'QUOTE_CONVENTION' in fm.columns else DIRECT_QUOTE
        ivv = to_usd(icv, fm['EXCHANGE_RATE'], quote_convention)

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
        target_pos = target_pos.where(has_price.loc[date], alpha_current_pos).astype(int)  # no price: hold
        trades_needed = target_pos - alpha_current_pos
        trades_needed = pd.to_numeric(trades_needed, errors='coerce').round().fillna(0).astype(int)

        # Buy at the close + half the spread, sell at the close - half the spread
        execution_price = px_closes + np.sign(trades_needed) * half_spread

        daily_instrument_pnls = []
        for ind, trade in trades_needed.items():
            if pd.isna(trade):
                daily_instrument_pnls.append(np.nan)
            else:
                # tick_value is in the instrument's own currency (e.g. EUR for RX1) - converted
                # to the portfolio's base currency (USD) here, same EXCHANGE_RATE conversion
                # already applied to ICV->IVV for sizing above. Previously this P&L was summed
                # into a USD AUM unconverted, as if 1 unit of local currency = 1 USD.
                quote_convention_ind = fm.loc[ind]['QUOTE_CONVENTION'] if 'QUOTE_CONVENTION' in fm.columns else DIRECT_QUOTE
                tick_value = to_usd(fm.loc[ind]['TICK_VALUE'], fm.loc[ind]['EXCHANGE_RATE'], quote_convention_ind)
                tick_size = fm.loc[ind]['TICK_SIZE']
                # Trading cost: spread paid on the fill + execution/clearing per contract
                commission_usd = to_usd(commission[ind], fm.loc[ind]['EXCHANGE_RATE'], quote_convention_ind)
                pnl_1 = ((px_closes[ind] - execution_price[ind]) * tick_value / tick_size * trade
                         - commission_usd * abs(trade))
                daily_instrument_pnls.append(pnl_1)

        daily_current_pnls = []
        for ind, cur_pos in current_pos.items():
            if not np.isnan(px_closes_prev[ind]):
                # Same currency conversion as above - tick_value -> portfolio base currency.
                quote_convention_ind = fm.loc[ind]['QUOTE_CONVENTION'] if 'QUOTE_CONVENTION' in fm.columns else DIRECT_QUOTE
                tick_value = to_usd(fm.loc[ind]['TICK_VALUE'], fm.loc[ind]['EXCHANGE_RATE'], quote_convention_ind)
                tick_size = fm.loc[ind]['TICK_SIZE']
                pnl_2 = (px_closes[ind] - px_closes_prev[ind]) * tick_value / tick_size * cur_pos
                daily_current_pnls.append(pnl_2)
            else:
                daily_current_pnls.append(0)

        # AUM updates with today's P&L here, after today's sizing decision is already locked
        # in above - this is what makes tomorrow's vol_scalar/cash_vol_tgt_daily reflect
        # today's result, i.e. AUM still compounds day to day exactly as before.
        # Both legs: mark-to-market on yesterday's position + today's trading costs.
        aum += np.nansum(daily_current_pnls) + np.nansum(daily_instrument_pnls)
        aums.append(aum)
        nav = aum.copy()
        navs.append(nav)

        px_closes_prev = px_closes

        values = [
            alpha_forecast, px_closes, one_perc_change, block_value, price_volatility,
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
        'px_close',  # back-adjusted PX_CLOSE_1D - the price the P&L is calculated from
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
    # - total_pnl_intraday_trades: trading costs of the day's trades (half spread + execution
    #   + clearing per contract) - always <= 0
    # - total_pnl_carried_forward: mark-to-market on positions held overnight (gross P&L)
    # - total_pnl_today: both together (net P&L) - what drives AUM
    instrument_trade_pnl_cols = [c for c in new_cols if c.endswith('_pnl_intraday_trades')]
    carried_forward_pnl_cols = [c for c in new_cols if c.endswith('_pnl_carried_forward')]
    trades_df['total_pnl_intraday_trades'] = trades_df[instrument_trade_pnl_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    trades_df['total_pnl_carried_forward'] = trades_df[carried_forward_pnl_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
    trades_df['total_pnl_today'] = trades_df['total_pnl_carried_forward'] + trades_df['total_pnl_intraday_trades']

    trades_df['AUM'] = aums
    trades_df['NAV'] = navs

    st.success("Forecast calculations complete!")
    st.write("Preview of generated trades/orders:")
    st.dataframe(trades_df.head(20))

    return trades_df
