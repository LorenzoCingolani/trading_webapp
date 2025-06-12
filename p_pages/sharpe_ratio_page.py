import streamlit as st
import os
import pandas as pd
import json
import numpy as np

def calculate_sharpe_forecast_returns(csvs_dictionary):
    results = []
    for inst, df in csvs_dictionary.items():
        if 'forecast*returns' in df.columns:
            series = df['forecast*returns'].dropna()
            if not series.empty and series.std() > 0:
                sharpe = series.mean() / series.std() * np.sqrt(252)
                results.append({'Instrument': inst, 'Sharpe Ratio': sharpe})
    return pd.DataFrame(results)

def run():
    st.title("Sharpe Ratio for forecast*returns")

    # Load framework_dict (instrument parameters)
    framework_json_path = os.path.join('DATA', 'output_instruments', 'control_output.json')
    with open(framework_json_path, 'r') as f:
        framework_dict = json.load(f)

    # Load instrument dataframes
    input_folder = os.path.join('DATA', 'output_instruments')
    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            inst = file.split('_')[0]
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[inst] = df

    # Show available instruments and columns
    st.subheader("Available Instruments and Columns")
    for inst, df in csvs_dictionary.items():
        st.write(f"**{inst}**: {list(df.columns)}")

    # Calculate Sharpe ratios for 'forecast*returns'
    sharpes_df = calculate_sharpe_forecast_returns(csvs_dictionary)

    # Show Sharpe ratios
    st.subheader("Sharpe Ratios for forecast*returns")
    st.dataframe(sharpes_df)

    # Optionally show the time series for each instrument
    st.subheader("Returns Time Series (first 10 rows per instrument)")
    for inst, df in csvs_dictionary.items():
        if 'forecast*returns' in df.columns:
            st.write(f"**{inst}**")
            st.dataframe(df['forecast*returns'].head(10))
            st.line_chart(df['forecast*returns'])

    # Download Sharpe ratios as CSV
    st.download_button(
        label="Download Sharpe Ratios CSV",
        data=sharpes_df.to_csv(index=False).encode('utf-8'),
        file_name='sharpe_ratios_forecast_returns.csv',
        mime='text/csv'
    )

    # Save the latest Sharpe ratios to JSON for future reference
    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    sharpes_dict_to_save = sharpes_df.to_dict(orient='records')
    with open(sharpe_results_path, 'w') as f:
        json.dump(sharpes_dict_to_save, f, indent=2)