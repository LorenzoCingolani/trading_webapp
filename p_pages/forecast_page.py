import streamlit as st
import os
import pandas as pd
import json
from steps.p5_framework_one_function import framework_main
from steps.p3_pdm import pdm_main

def run():
    st.title("Forecast Generation")

    json_path = os.path.join('DATA', 'output_instruments', 'control_output.json')
    input_folder = os.path.join('DATA', 'input_instruments')
    forecast_folder = os.path.join('DATA', 'combinedForecast')
    output_path = os.path.join('DATA', 'order_folder', 'orders_new778.csv')

    # load control data
    with open(json_path, 'r') as f:
        control = json.load(f)

    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[file[:-4]] = df

    aum = 10_000_000
    PDM = pdm_main(control, csvs_dictionary)
    
    st.write("PDM calculated successfully. its values are:", PDM)
    date_format = "%d/%m/%Y"

    order_file = framework_main(control, forecast_folder, csvs_dictionary, PDM, date_format, aum, is_markov=False)
    order_file.to_csv(output_path)

    st.success("Forecasting and order file generated!")
    st.dataframe(order_file)
