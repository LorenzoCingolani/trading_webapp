import streamlit as st
import os
import pandas as pd
import json
from trading.p1_analysis import main_analysis

def run():
    st.title("Main Analysis")
    st.write("Running main analysis on all input instruments...")

    input_folder = os.path.join('DATA', 'input_instruments')
    json_path = os.path.join('DATA', 'input_main', 'input_main.json')
    csvs_dictionary = {}

    with open(json_path, 'r') as f:
        control = json.load(f)

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))
            name = file[:-4]
            csvs_dictionary[name] = df

            if name in control:
                control[name].update({
                    'INSTRUMENT': name,
                    'CURRENCY': df['CRNCY'].iloc[0],
                    'EXCHANGE': df['EXCHANGE'].iloc[0],
                    'SECTYPE': df['SECTYPE'].iloc[0],
                    'TICK_SIZE': df['TICK_SIZE'].iloc[0],
                    'TICK_VALUE': df['TICK_VALUE'].iloc[0],
                    'POINT_VALUE': df['POINT_VALUE'].iloc[0],
                    'CONTRACT_VALUE': df['CONTRACT_VALUE'].iloc[0],
                    'EXCHANGE_RATE': df['Exchange rate'].iloc[0],
                    'STANDARD_COST': df['Standard Cost'].iloc[0],
                })

    main_analysis(control, csvs_dictionary, input_folder)
    st.success("Main analysis complete.")
