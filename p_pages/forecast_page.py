import streamlit as st
import os
import pandas as pd
from steps.p5_framework_one_function import framework_main
from steps.p3_pdm import pdm_main

def run():
    st.title("Forecast Generation")

    if 'forecast_run' not in st.session_state:
        st.session_state.forecast_run = False

    if st.button("Run forecast", key="run_forecast"):
        st.session_state.forecast_run = True

    if not st.session_state.forecast_run:
        st.info("Press Run forecast to generate forecasts and order files.")
        return

    csv_path = os.path.join('DATA', 'output_instruments', 'control_output.csv')
    json_path = os.path.join('DATA', 'output_instruments', 'control_output.json')
    input_folder = os.path.join('DATA', 'input_instruments')
    forecast_folder = os.path.join('DATA', 'combinedForecast')
    output_path = os.path.join('DATA', 'order_folder', 'orders_time.csv')
    output_path = output_path.replace('time', pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'))
    st.info(f"Output path: {output_path}")

    if os.path.exists(csv_path):
        control_df = pd.read_csv(csv_path)
        control = {}
        for _, row in control_df.iterrows():
            instrument = row['INSTRUMENT']
            values = row.drop(labels=['INSTRUMENT']).to_dict()
            values['INSTRUMENT'] = instrument
            control[instrument] = values
    elif os.path.exists(json_path):
        try:
            control_df = pd.read_json(json_path, orient='index')
            control_df = control_df.reset_index().rename(columns={'index': 'INSTRUMENT'})
            control = {}
            for _, row in control_df.iterrows():
                instrument = row['INSTRUMENT']
                control[instrument] = row.drop(labels=['INSTRUMENT']).to_dict()
        except Exception as e:
            st.error(f"Failed to read legacy JSON control file: {e}")
            return
    else:
        st.error('No control output found. Expected one of:\n  DATA/output_instruments/control_output.csv\n  DATA/output_instruments/control_output.json')
        return

    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[file[:-4]] = df

    aum = 10_000_000
    PDM = pdm_main(control, csvs_dictionary)

    st.write("PDM calculated successfully. its value is:", PDM)
    date_format = "%d/%m/%Y"

    order_file = framework_main(control, forecast_folder, csvs_dictionary, PDM, date_format, aum, is_markov=False)
    order_file.to_csv(output_path)

    st.success("Forecasting and order file generated!")
    st.dataframe(order_file)

    if st.button("Run forecast again", key="rerun_forecast"):
        st.session_state.forecast_run = False
        st.experimental_rerun()
