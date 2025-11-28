import streamlit as st
import os
import pandas as pd
from steps.p3_pdm import pdm_main

def run():
    st.title("PDM")
    input_folder = os.path.join('DATA', 'input_instruments')
    csv_path = os.path.join('DATA', 'input_main', 'input_main.csv')

    csvs_dictionary = {}

    # Load control data from CSV and convert to dict format
    control_df = pd.read_csv(csv_path)
    control = {}
    for _, row in control_df.iterrows():
        instrument = row['INSTRUMENT']
        control[instrument] = {'INSTRUMENT_WEIGHTS': row['INSTRUMENT_WEIGHTS'], 'INSTRUMENT': instrument}

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[file[:-4]] = df

    pdm_result = pdm_main(control, csvs_dictionary)
    st.success("PDM process complete.")
    st.write("PDM result:", pdm_result)
    st.metric("Portfolio Diversification Multiplier (PDM)", f"{pdm_result:.4f}")
