import streamlit as st
import os
import pandas as pd
import json
from steps.p1_analysis import main_analysis
import shutil

def run():
    st.title("Main Analysis")
    st.write("Running main analysis on all input instruments...")

    input_folder = os.path.join('DATA', 'input_instruments')


    # %%
    ### REMOVE FOLDERS IF THEY EXIST
    # remove output_instruments folder if it exists
    output_folder = os.path.join('DATA', 'output_instruments')
    if os.path.exists(output_folder): ## just to remove the folder if it exists
        shutil.rmtree(output_folder)
    # remove the combined forecast folder if it exists
    combined_forecast_folder = os.path.join('DATA', 'combined_forecast')
    if os.path.exists(combined_forecast_folder):
        shutil.rmtree(combined_forecast_folder)
    # remove order_folder if it exists
    order_folder = os.path.join('DATA', 'order_folder')
    if os.path.exists(order_folder):
        shutil.rmtree(order_folder)
    

    # create these folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(combined_forecast_folder, exist_ok=True)
    os.makedirs(order_folder, exist_ok=True)

    # %%
    
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

    # Show control data sample
    with st.expander("Show control (framework) data sample"):
        st.json({k: control[k] for k in list(control.keys())[:3]})  # show first 3 instruments

    # Show csvs_dictionary data sample
    with st.expander("Show csvs_dictionary (input CSVs) sample"):
        for k in list(csvs_dictionary.keys())[:3]:  # show first 3 instruments
            st.write(f"Instrument: {k}")
            st.dataframe(csvs_dictionary[k].head())

    main_analysis(control, csvs_dictionary)
    st.success("Main analysis complete.")

    # Save control variable to DATA/output_instruments
    output_folder = os.path.join('DATA', 'output_instruments')
    
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'control_output.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types have .item() method
            return obj.item()
        else:
            return obj
    
    control_serializable = convert_numpy_types(control)
    
    with open(output_path, 'w') as f:
        json.dump(control_serializable, f, indent=4)
    st.info(f"Control data saved to {output_path}")

# To load the saved control variable later:
# with open('DATA/output_instruments/control_output.json', 'r') as f:
#     control_loaded = json.load(f)