import streamlit as st
import os
import pandas as pd
import json

from steps.p6_sharpe_ratio import run_sharpe_ratio_page

def run():
    st.title("Sharpe Ratio Page")

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

    # Show available instruments and their columns
    st.subheader("Available Instruments and Columns")
    for inst, df in csvs_dictionary.items():
        st.write(f"**{inst}**: {list(df.columns)}")

    # Show framework_dict as a dataframe for transparency
    st.subheader("Instrument Weights and Parameters")
    st.dataframe(pd.DataFrame(framework_dict).T)

    # Run the detailed Sharpe Ratio analysis page
    run_sharpe_ratio_page(csvs_dictionary, framework_dict)

    # save the Sharpe Ratio results to a file
    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    if os.path.exists(sharpe_results_path):
        with open(sharpe_results_path, 'r') as f:
            sharpe_results = json.load(f)
        st.subheader("Saved Sharpe Ratio Results")
        st.json(sharpe_results)
    else:
        st.warning("No saved Sharpe Ratio results found.")
    st.success("Sharpe Ratio analysis completed successfully.")

    # save dataframes to CSV files
    for inst, df in csvs_dictionary.items():
        df.to_csv(os.path.join('DATA', 'output_instruments', f"{inst}__sharpe_results.csv"), index=False)