import streamlit as st
import os
import json
import pandas as pd
from steps.p3_pdm import pdm_main

def run():
    st.title("PDM")
    input_folder = os.path.join('DATA', 'input_instruments')
    json_path = os.path.join('DATA', 'input_main', 'input_main.json')

    csvs_dictionary = {}

    with open(json_path, 'r') as f:
        control = json.load(f)

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[file[:-4]] = df

    pdm_result = pdm_main(control, csvs_dictionary)
    st.success("PDM process complete.")
    st.write("PDM result:", pdm_result)
    st.metric("Portfolio Diversification Multiplier (PDM)", f"{pdm_result:.4f}")
