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

    # Show available instruments and their date counts
    st.write("Instrument data lengths:")
    for k, v in all_alpha_forecast.items():
        st.write(f"{k}: {len(v)} rows")

    # Find the union of all dates across all instruments
    all_dates = set()
    for series in all_alpha_forecast.values():
        all_dates.update(series.index)
    all_dates = sorted(list(all_dates))
    st.write(f"Total unique dates across all instruments: {len(all_dates)}")

    # Reindex all series to the union of dates, so missing data is visible as NaN
    for k in all_alpha_forecast:
        all_alpha_forecast[k] = all_alpha_forecast[k].reindex(all_dates)
        all_px_closes[k] = all_px_closes[k].reindex(all_dates)
        all_std_dev[k] = all_std_dev[k].reindex(all_dates)

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

    # For debugging: store daily summaries
    daily_summary = []

    for idx, date in enumerate(date_list):
        alpha_forecast = alpha_forecast_df.loc[date].astype(float)
        px_closes = px_close_df.loc[date].astype(float)
        std_dev = std_dev_df.loc[date].astype(float)
        cash_vol_tgt_daily = [aum * 0.2 / math.sqrt(256)] * len(alpha_forecast)

        one_perc_change = px_closes * 0.01
        block_value = one_perc_change * fm['POINT_VALUE']
        price_volatility = np.round((std_dev / px_closes) * 100, 2)
        icv = price_volatility * block_value
        ivv = icv * fm['EXCHANGE_RATE']
        vol_scalar = cash_vol_tgt_daily / ivv
        pos_contracts = vol_scalar * alpha_forecast / 10
        target_pos = pos_contracts * PDM * fm['INSTRUMENT_WEIGHTS']
        trades_needed = target_pos - alpha_current_pos 
        
        # Show all variable values as a table for this date
                # Show all variable values as a table for this date, with formulas as column names
        # if last line then show
        if idx in [0,1,2,3,4]:
            st.write("Final date reached, showing all variable values for this date.")
            details_df = pd.DataFrame({
                "px_closes": px_closes,
                "std_dev": std_dev,
                "cash_vol_tgt_daily = aum * 0.2 / sqrt(256)": cash_vol_tgt_daily,
                "one_perc_change = px_closes * 0.01": one_perc_change,
                "block_value = one_perc_change * POINT_VALUE": block_value,
                "price_volatility = (std_dev / px_closes) * 100": price_volatility,
                "icv = price_volatility * block_value": icv,
                "ivv = icv * EXCHANGE_RATE": ivv,
                "vol_scalar = cash_vol_tgt_daily / ivv": vol_scalar,
                "alpha_forecast": alpha_forecast,
                "pos_contracts = vol_scalar * alpha_forecast / 10": pos_contracts,
                "target_pos = pos_contracts * PDM * INSTRUMENT_WEIGHTS": target_pos,
                "alpha_current_pos": alpha_current_pos,
                "trades_needed = target_pos - alpha_current_pos": trades_needed,
            })
            details_df.index.name = "Instrument"
            st.subheader(f"Details for {date.date()}")
            st.dataframe(details_df)


        daily_instrument_pnls = []
        missing_instruments = []
        for ind, trade in trades_needed.items():
            if pd.isna(trade):
                daily_instrument_pnls.append(np.nan)
                missing_instruments.append(ind)
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

        # Debug: show missing instruments for this date
        if missing_instruments:
            st.warning(f"Date {date.date()}: Missing data for instruments: {missing_instruments}")

        # Sum PnL across all instruments for this day
        total_daily_pnl = np.nansum(daily_instrument_pnls)
        prev_aum = aum
        aum += total_daily_pnl
        aums.append(aum)

        # Store daily summary for debugging
        daily_summary.append({
            "date": date,
            "prev_aum": prev_aum,
            "total_daily_pnl": total_daily_pnl,
            'total_current_pnl': np.nansum(daily_current_pnls),
            "aum": aum,

            "missing_instruments": missing_instruments,
            "num_instruments": len(trades_needed),
            "num_missing": len(missing_instruments)
        })

        values = [
            alpha_forecast, one_perc_change, block_value, price_volatility,
            cash_vol_tgt_daily, std_dev, icv, ivv, vol_scalar,
            pos_contracts, trades_needed, target_pos,
            daily_instrument_pnls, daily_current_pnls
        ]

        px_closes_prev = px_closes
        output = []
        ret = np.array([list(val) for val in values])
        for j in range(ret.shape[1]):
            for i in range(ret.shape[0]):
                output.append(ret[i][j])
        trades.append(output)

        alpha_current_pos = pd.Series(np.nan_to_num(target_pos), index=fm.index)
        current_pos = pd.Series(np.nan_to_num(target_pos), index=fm.index)
        progress_bar2.progress((idx + 1) / len(date_list))

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

    st.success("Forecast calculations complete!")

    # Show daily summary table for debugging
    st.subheader("Daily P&L and AUM Summary (first 10 days)")
    st.dataframe(pd.DataFrame(daily_summary).head(10))

    # Show a filtered view for convenience, but do not overwrite trades_df
    st.markdown("### Trades DataFrame (filtered view)")
    selected_columns = st.multiselect(
        "Select columns to display",
        options=trades_df.columns.tolist(),
        default=trades_df.columns.tolist()[:10]  # Default to first 10 columns
    )
    if selected_columns:
        st.dataframe(trades_df[selected_columns])
    else:
        st.warning("No columns selected. Displaying all columns.")
        st.dataframe(trades_df)

    st.markdown("### Full Trades DataFrame (all columns, for download or further use)")
    st.dataframe(trades_df)

    return trades_df