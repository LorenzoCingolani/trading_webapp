import streamlit as st
import os
import pandas as pd
import json
import numpy as np

from steps.p6_sharpe_ratio import run_sharpe_ratio_page

def calculate_sharp_ratio_forcast_times_return(csvs_dictionary, framework_dict):
    # For each instrument, calculate Sharpe for every numeric column
    sharpes = {}
    returns_dict = {}
    weights = {}
    all_sharpes_df = []
    for inst, df in csvs_dictionary.items():
        inst_sharpes = {}
        weights[inst] = framework_dict[inst]['INSTRUMENT_WEIGHTS'] if inst in framework_dict else 1.0
        for col in ['forecast*returns']:
            returns = df[col].dropna()
            if not returns.empty and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                inst_sharpes[col] = sharpe
                # Save returns for the first valid column (for portfolio calc)
                if inst not in returns_dict:
                    returns_dict[inst] = returns
            else:
                inst_sharpes[col] = np.nan
        sharpes[inst] = inst_sharpes
        # For display as dataframe
        for col, val in inst_sharpes.items():
            all_sharpes_df.append({'Instrument': inst, 'Column': col, 'Sharpe Ratio': val})
    sharpes_df = pd.DataFrame(all_sharpes_df)
    return sharpes, returns_dict, weights, sharpes_df

def calculate_portfolio_sharpe(returns_dict, weights):
    df = pd.DataFrame(returns_dict)
    df = df.dropna()
    if df.empty:
        return np.nan, None, None
    w = np.array([weights[inst] for inst in df.columns])
    w = w / w.sum()
    port_ret = df.values @ w
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    return sharpe, port_ret, df.index

def run():
    st.title("Sharpe Ratio Page")

    # Load framework_dict (instrument parameters)
    framework_json_path = os.path.join('DATA', 'output_instruments', 'control_output.json')
    with open(framework_json_path, 'r') as f:
        framework_dict = json.load(f)

    # Load instrument dataframes
    input_folder = os.path.join('DATA', 'output_instruments')
    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            inst = file.split('_')[0]
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary[inst] = df

    # Show available instruments and their columns
    st.subheader("Available Instruments and Columns")
    for inst, df in csvs_dictionary.items():
        # Make instrument name bold and columns gray
        columns_html = " ".join([f"<span style='color:gray; background-color:#f0f0f0; border-radius:4px; padding:2px 6px; margin-right:4px;'>{col}</span>" for col in df.columns])
        st.markdown(f"**{inst}**: {columns_html}", unsafe_allow_html=True)
 
    # Show framework_dict as a dataframe for transparency
    st.subheader("Instrument Weights and Parameters")
    st.dataframe(pd.DataFrame(framework_dict).T)

    # Run the detailed Sharpe Ratio analysis page
    run_sharpe_ratio_page(csvs_dictionary, framework_dict)

    # save the Sharpe Ratio results to a file
    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    if os.path.exists(sharpe_results_path):
        with open(sharpe_results_path, 'r') as f:
            sharpe_results = json.load(f)
        st.subheader("Saved Sharpe Ratio Results")
        st.json(sharpe_results)
    else:
        st.warning("No saved Sharpe Ratio results found.")
    st.success("Sharpe Ratio analysis completed successfully.")

    # save dataframes to CSV files
    for inst, df in csvs_dictionary.items():
        df.to_csv(os.path.join('DATA', 'output_instruments', f"{inst}__sharpe_results.csv"), index=False)

    # Calculate Sharpe ratios for all numeric columns of all instruments
    sharpes, returns_dict, weights, sharpes_df = calculate_sharp_ratio_forcast_times_return(csvs_dictionary, framework_dict)

    # Show all Sharpe ratios in a dataframe
    st.subheader("Sharpe Ratios for All Instruments and Columns")
    st.dataframe(sharpes_df)

    # Show as pivot table for easier reading
    st.subheader("Sharpe Ratios Pivot Table")
    if not sharpes_df.empty:
        pivot = sharpes_df.pivot(index='Instrument', columns='Column', values='Sharpe Ratio')
        st.dataframe(pivot)

    # Show returns time series for each instrument (first numeric column)
    st.subheader("Returns Time Series (first 10 rows per instrument)")
    for inst, returns in returns_dict.items():
        st.write(f"**{inst}**")
        st.dataframe(returns.head(10))
        st.line_chart(returns.rename(f"{inst} returns"))

    # Portfolio Sharpe ratio and plots
    st.subheader("Portfolio Sharpe Ratio (using first valid column per instrument)")
    port_sharpe, port_ret, port_dates = calculate_portfolio_sharpe(returns_dict, weights)
    st.write(port_sharpe)

    if port_ret is not None:
        st.subheader("Portfolio Cumulative Returns")
        port_cum = np.cumsum(port_ret)
        port_cum_df = pd.DataFrame({'Cumulative Return': port_cum}, index=port_dates)
        st.line_chart(port_cum_df)

        # Show correlation matrix between instruments
        st.subheader("Correlation Matrix of Returns")
        returns_df = pd.DataFrame(returns_dict)
        st.dataframe(returns_df.corr())

    # Download Sharpe ratios as CSV
    st.download_button(
        label="Download Sharpe Ratios CSV",
        data=sharpes_df.to_csv(index=False).encode('utf-8'),
        file_name='sharpe_ratios_all.csv',
        mime='text/csv'
    )

    # Save the latest Sharpe ratios to JSON for future reference
    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    sharpes_dict_to_save = sharpes_df.to_dict(orient='records')
    with open(sharpe_results_path, 'w') as f:
        json.dump(sharpes_dict_to_save, f, indent=2)