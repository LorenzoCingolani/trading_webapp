import streamlit as st
import os
import json
from steps.p2_validation import validation_main
import pandas as pd

def run():
    st.title("Validation")
    validation_input_folder = os.path.join('DATA', 'output_instruments')

    if 'validation_run' not in st.session_state:
        st.session_state.validation_run = False

    if st.button("Run validation", key="run_validation"):
        st.session_state.validation_run = True

    input_folder = os.path.join('DATA', 'input_instruments')
    instrument_names = [file[:-4] for file in os.listdir(input_folder) if file.endswith('.csv')]

    sample_size = st.number_input("Sample size for validation (default -1 for all data)", min_value=-1, value=-1, step=1)
    st.info(f"Using sample size: {sample_size}")

    with st.expander("Show instrument names"):
        st.write(instrument_names)

    with st.expander("Show validation input folder"):
        st.write(validation_input_folder)

    if not st.session_state.validation_run:
        st.info("Press Run validation to execute the validation process.")
        return

    control_csv = os.path.join('DATA', 'output_instruments', 'control_output.csv')
    control_json = os.path.join('DATA', 'output_instruments', 'control_output.json')

    if os.path.exists(control_csv):
        control_df = pd.read_csv(control_csv)
        control = {}
        for _, row in control_df.iterrows():
            instrument = row['INSTRUMENT']
            values = row.drop(labels=['INSTRUMENT']).to_dict()
            values['INSTRUMENT'] = instrument
            control[instrument] = values
    elif os.path.exists(control_json):
        with open(control_json, 'r') as f:
            control = json.load(f)
    else:
        st.error(
            'No control output found. Expected one of:\n'
            '  DATA/output_instruments/control_output.csv\n'
            '  DATA/output_instruments/control_output.json'
        )
        return

    with st.expander("Show control (framework) data sample"):
        st.json({k: control[k] for k in list(control.keys())[:3]})

    output_files = [f for f in os.listdir(validation_input_folder) if f.endswith('.csv')]
    if output_files:
        with st.expander("Show sample output data (first file)"):
            df = pd.read_csv(os.path.join(validation_input_folder, output_files[0]))
            st.write(f"File: {output_files[0]}")
            st.header(f"rows {df.shape[0]} columns {df.shape[1]}")
            st.dataframe(df.head())

    st.info("Calling validation_main with:")
    st.code(f"validation_main(instrument_names, control, {sample_size}, '{validation_input_folder}')")

    validation_main(instrument_names, control, sample_size, validation_input_folder)
    st.success("Validation complete.")

    if st.button("Run validation again", key="rerun_validation"):
        st.session_state.validation_run = False
        st.experimental_rerun()
