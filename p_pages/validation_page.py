import streamlit as st
import os
import json
from steps.p2_validation import validation_main
import pandas as pd

def run():
    st.title("Validation")
    validation_input_folder = os.path.join('DATA', 'output_instruments')

    #To load the saved control variable later:
    with open('DATA/output_instruments/control_output.json', 'r') as f:
        control = json.load(f)

    instrument_names = []

    

    input_folder = os.path.join('DATA', 'input_instruments')
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            instrument_names.append(file[:-4])

    sample_size = -1  # -1 means use all available data
    # by defaul else write using st
    st.write("Sample size for validation: -1 (use all available data)")
    if 'sample_size' in st.session_state:
        sample_size = st.session_state.sample_size
        st.write(f"Sample size for validation: {sample_size} (from session state)")
    else:
        st.session_state.sample_size = sample_size
        st.write("Sample size for validation: -1 (default, use all available data)")

    # Show variables and data samples
    
    with st.expander("Show instrument names"):
        st.write(instrument_names)

    with st.expander("Show control (framework) data sample"):
        st.json({k: control[k] for k in list(control.keys())[:3]})  # show first 3 instruments

    with st.expander("Show validation input folder"):
        st.write(validation_input_folder)

    with st.expander("Show sample size used for validation"):
        st.write(sample_size)

    # Optionally, show a sample of output data if available
    output_files = [f for f in os.listdir(validation_input_folder) if f.endswith('.csv')]
    if output_files:
        with st.expander("Show sample output data (first file)"):
            df = pd.read_csv(os.path.join(validation_input_folder, output_files[0]))
            st.write(f"File: {output_files[0]}")
            st.header("rows {df.shape[0]} columns {df.shape[1]}")
            st.dataframe(df.head())

    st.info("Calling validation_main with:")
    st.code(f"validation_main(instrument_names, control, {sample_size}, '{validation_input_folder}')")

    validation_main(instrument_names, control, sample_size, validation_input_folder)
    st.success("Validation complete.")