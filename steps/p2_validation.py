import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import streamlit as st

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
    for ins_name in inst_names:
        st.subheader(f"Instrument: {ins_name}")
        commodity_parameters = control_dictionary.get(ins_name)
        if not commodity_parameters:
            st.warning(f"No parameters found for {ins_name}")
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

        for day in val_days.tolist():
            commodity_subset = [df[df['Date'] < day] for df in commodity_data.values()]
            forecasted_value = forecast(commodity_subset, Weights)
            validation_data.append((day, *forecasted_value))

        output = pd.DataFrame(validation_data, columns=['Date', 'FinalForecast', 'Multiplier'])

        for key, data in commodity_data.items():
            output[f'{key}_forecast'] = data[data['Date'] >= start_date]['capped_forecast'].values

        output_path = os.path.join(output_dir, f'{PrCode}.csv')
        output.to_csv(output_path, index=False)
        st.success(f"Saved forecast to: {output_path}")

        # Show a preview of the output
        with st.expander(f"Show combined forecast output for {ins_name}"):
            st.dataframe(output.head(10))