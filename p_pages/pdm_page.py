import streamlit as st
import os
import pandas as pd
from steps.p3_pdm import pdm_main

def run():
    st.title("PDM")
    input_folder = os.path.join('DATA', 'input_instruments')
    csv_path = os.path.join('DATA', 'input_main', 'input_main.csv')

    if 'pdm_run' not in st.session_state:
        st.session_state.pdm_run = False

    if st.button("Run PDM", key="run_pdm"):
        st.session_state.pdm_run = True

    if not st.session_state.pdm_run:
        st.info("Press Run PDM to calculate the portfolio diversification multiplier.")
        return

    csvs_dictionary = {}

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

    if st.button("Run PDM again", key="rerun_pdm"):
        st.session_state.pdm_run = False
        st.experimental_rerun()
