import os
import pandas as pd
import numpy as np
import time
from datetime import timedelta, datetime
import streamlit as st


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
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

def load_commodity_data(commodity: str, CsvFolder: str) -> dict:
    all_data = {}
    all_output_files = os.listdir(CsvFolder)
    st.info(f'All output files: {all_output_files}')

    for filename in all_output_files:
        if filename.startswith(commodity) and filename.endswith('.csv'):
            st.write(f"Loading file: {filename}")
            data = pd.read_csv(os.path.join(CsvFolder, filename))
            data.dropna(subset=['Date'], inplace=True)

            try:
                data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
            except ValueError:
                data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=True)

            model_name = filename.replace(f'{commodity}_', '').replace('.csv', '')
            all_data[model_name] = data

    return all_data

def forecast(commodity_data: list[pd.DataFrame], Weights: np.ndarray) -> tuple[float, float]:
    CumList = [data['forecast*returns'].values for data in commodity_data]
    CorrMat = pd.DataFrame(CumList).T.corr()

    M = min(1. / np.sqrt(np.dot(Weights.T, np.dot(CorrMat, Weights))), 2.5)
    CapForecastList = [data['capped_forecast'].iloc[-1] for data in commodity_data]

    UnweightedForecast = np.dot(Weights, CapForecastList)
    FinalForecast = M * UnweightedForecast

    return FinalForecast, M

def validation_main(inst_names: list[str],
                    control_dictionary: dict,
                    validation_days: int,
                    CsvFolder: str) -> None:
    output_dir = os.path.join('DATA', 'combinedForecast')
    os.makedirs(output_dir, exist_ok=True)
    # remove old files
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

    st.header("Validation Progress")
    overall_start = time.time()
    total_instruments = len(inst_names)
    overall_bar = st.progress(0.0)
    overall_status = st.empty()
    current_status = st.empty()

    for inst_idx, ins_name in enumerate(inst_names, start=1):
        overall_status.info(_progress_text("Overall instruments", inst_idx - 1, total_instruments, overall_start))
        st.subheader(f"Instrument: {ins_name}")
        commodity_parameters = control_dictionary.get(ins_name)
        if not commodity_parameters:
            st.warning(f"No parameters found for {ins_name}")
            overall_bar.progress(inst_idx / total_instruments if total_instruments else 1.0)
            continue

        PrCode = commodity_parameters['INSTRUMENT']
        st.write(f"Processing {ins_name} ({PrCode})")

        commodity_data = load_commodity_data(PrCode, CsvFolder)
        NModels = len(commodity_data)

        if NModels == 0:
            st.warning(f"No models found for {PrCode}")
            continue

        st.write(f"Number of models found: {NModels}")
        st.write(f"Model names: {list(commodity_data.keys())}")

        try:
            Weights = np.ones(NModels) / NModels
        except ZeroDivisionError:
            st.error(f"ZeroDivisionError: No models for {PrCode}")
            continue

        model1 = list(commodity_data.values())[0]
        start_date = (
            model1['Date'].iloc[1]
            if validation_days == -1
            else model1['Date'].max() - timedelta(days=validation_days)
        )

        val_days = model1[model1['Date'] >= start_date]['Date']
        validation_data = []

        days_list = val_days.tolist()
        day_count = len(days_list)
        instrument_start = time.time()
        day_bar = st.progress(0.0)
        day_status = st.empty()

        for day_idx, day in enumerate(days_list, start=1):
            commodity_subset = [df[df['Date'] < day] for df in commodity_data.values()]
            forecasted_value = forecast(commodity_subset, Weights)
            validation_data.append((day, *forecasted_value))
            if day_idx == 1 or day_idx == day_count or day_idx % 25 == 0:
                day_bar.progress(day_idx / day_count if day_count else 1.0)
                day_status.info(_progress_text(f"{ins_name} validation days", day_idx, day_count, instrument_start))

        day_bar.progress(1.0)
        day_status.success(_progress_text(f"{ins_name} validation days", day_count, day_count, instrument_start))

        output = pd.DataFrame(validation_data, columns=['Date', 'FinalForecast', 'Multiplier'])

        for key, data in commodity_data.items():
            output[f'{key}_forecast'] = data[data['Date'] >= start_date]['capped_forecast'].values

        output_path = os.path.join(output_dir, f'{PrCode}.csv')
        output.to_csv(output_path, index=False)
        st.success(f"Saved forecast to: {output_path}")
        # SHOW OUTPUT with header
        
        # Show a preview of the output
        st.header(f"Combined Forecast Output for {ins_name} rows {output.shape[0]} columns {output.shape[1]}")
        with st.expander(f"Show combined forecast output for {ins_name}"):
            st.dataframe(output)

        overall_bar.progress(inst_idx / total_instruments if total_instruments else 1.0)
        overall_status.info(_progress_text("Overall instruments", inst_idx, total_instruments, overall_start))
        current_status.success(f"Finished {ins_name} in {_format_seconds(time.time() - instrument_start)}")

    overall_bar.progress(1.0)
    overall_status.success(_progress_text("Overall instruments", total_instruments, total_instruments, overall_start))
