import streamlit as st
import os
import json
from steps.p2_validation import validation_main
import pandas as pd

def run():
    st.title("Validation")
    validation_input_folder = os.path.join('DATA', 'output_instruments')
    json_path = os.path.join('DATA', 'input_main', 'input_main.json')
    instrument_names = []

    with open(json_path, 'r') as f:
        control = json.load(f)

    input_folder = os.path.join('DATA', 'input_instruments')
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            instrument_names.append(file[:-4])

    validation_main(instrument_names, control, 100, validation_input_folder)
    st.success("Validation complete.")
