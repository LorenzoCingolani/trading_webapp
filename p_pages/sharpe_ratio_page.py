import streamlit as st
import os
import pandas as pd
import json
import numpy as np

def calculate_sharpe_forecast_returns(csvs_dictionary):
    results = []
    for inst_version, df in csvs_dictionary.items():
        if 'forecast*returns' in df.columns:
            series = df['forecast*returns'].dropna()
            if not series.empty and series.std() > 0:
                sharpe = series.mean() / series.std() * np.sqrt(252)
                # Split inst_version into instrument and version
                if '_' in inst_version:
                    instrument, version = inst_version.split('_', 1)
                    if version == 'results':
                        continue  # Skip results version
                else:
                    instrument, version = inst_version, ''
                results.append({
                    'Instrument': instrument,
                    'Version': version,
                    'Sharpe Ratio': sharpe
                })
    return pd.DataFrame(results)

def run():
    st.title("Sharpe Ratio for forecast*returns")

    st.write("This page calculates the Sharpe ratio for the 'forecast*returns' column in each instrument's dataframe Each Strategy.")

    # Load instrument dataframes
    input_folder = os.path.join('DATA', 'output_instruments')
    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            inst = file.split('_')[0]
            version = file.split('_')[-1].replace('.csv', '')
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary.setdefault(inst, {})[version] = df

    # Calculate Sharpe ratios for 'forecast*returns'
    sharpes = []
    for inst, versions in csvs_dictionary.items():
        for version, df in versions.items():
            if version == "results":
                continue  # Skip 'results' version
            if 'forecast*returns' in df.columns:
                series = df['forecast*returns'].dropna()
                if not series.empty and series.std() > 0:
                    sharpe = series.mean() / series.std() * np.sqrt(252)
                    sharpes.append({'Instrument': inst, 'Version': version, 'Sharpe Ratio': sharpe})
    sharpes_df = pd.DataFrame(sharpes)

    st.subheader("Sharpe Ratios for forecast*returns")
    st.dataframe(sharpes_df)

    # Dropdown to select instrument
    instruments = list(csvs_dictionary.keys())
    selected_inst = st.selectbox("Select Instrument", instruments)

    # Get versions for selected instrument
    versions = [v for v in csvs_dictionary[selected_inst].keys() if v != "results"]
    n_versions = len(versions)
    default_weight = 1.0 / n_versions

    st.subheader(f"Set Weights for {selected_inst} Versions (sum ≤ 1.0)")
    weights = []
    total_weight = 0.0
    for version in versions:
        weight = st.number_input(
            f"Weight for {version}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.01,
            key=f"{selected_inst}_{version}_weight"
        )
        weights.append(weight)
        total_weight += weight

    # Normalize weights if sum > 1
    if total_weight > 1.0:
        st.warning("Total weight exceeds 1.0. Weights will be normalized.")
        weights = [w / total_weight for w in weights]
        total_weight = sum(weights)

    # Show weighted Sharpe ratios
    st.subheader("Weighted Sharpe Ratios")
    weighted_sharpes = []
    sum_weighted_sharpe = 0.0
    for version, weight in zip(versions, weights):
        sharpe_row = sharpes_df[(sharpes_df['Instrument'] == selected_inst) & (sharpes_df['Version'] == version)]
        if not sharpe_row.empty:
            sharpe = sharpe_row['Sharpe Ratio'].values[0]
            weighted = sharpe * weight
            weighted_sharpes.append({'Version': version, 'Weight': weight, 'Sharpe Ratio': sharpe, 'Weighted Sharpe': weighted})
            sum_weighted_sharpe += weighted

    st.dataframe(pd.DataFrame(weighted_sharpes))
    st.write(f"**Sum of Weighted Sharpe Ratios:** {sum_weighted_sharpe:.4f}")

    # Show time series for each version in tabs
    st.subheader(f"Returns Time Series for {selected_inst} (first 10 rows per version)")
    tab_labels = versions
    tabs = st.tabs(tab_labels)
    for i, version in enumerate(versions):
        df = csvs_dictionary[selected_inst][version]
        with tabs[i]:
            if 'forecast*returns' in df.columns:
                st.write(f"**{selected_inst} - {version}**")
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

