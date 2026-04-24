import streamlit as st
import os
import pandas as pd
import json
import numpy as np

def calculate_sharpe_forecast_returns(csvs_dictionary):
    results = []
    for inst_version, df in csvs_dictionary.items():
        if 'forecast*returns' in df.columns:
            series = df['forecast*returns'].dropna()
            if not series.empty and series.std() > 0:
                sharpe = series.mean() / series.std() * np.sqrt(252)
                # Split inst_version into instrument and version
                if '_' in inst_version:
                    instrument, version = inst_version.split('_', 1)
                    if version == 'results':
                        continue  # Skip results version
                else:
                    instrument, version = inst_version, ''
                results.append({
                    'Instrument': instrument,
                    'Version': version,
                    'Sharpe Ratio': sharpe
                })
    return pd.DataFrame(results)

def run():
    st.title("Sharpe Ratio for forecast*returns")
    st.write("This page calculates the Sharpe ratio for the 'forecast*returns' column in each instrument's dataframe Each Strategy.")

    if 'sharpe_started' not in st.session_state:
        st.session_state.sharpe_started = False
    if 'sharpe_done' not in st.session_state:
        st.session_state.sharpe_done = False
    if 'sharpe_results' not in st.session_state:
        st.session_state.sharpe_results = {}

    if st.session_state.sharpe_done:
        st.success("Sharpe analysis already completed. Use Run Sharpe analysis again to rerun.")
        results = st.session_state.sharpe_results
        if results:
            st.subheader("Sharpe ratios")
            st.dataframe(results.get("sharpes_df", []))
            st.write("Saved results to DATA/output_instruments/sharpe_results.json")
        if st.button("Run Sharpe analysis again", key="rerun_sharpe"):
            st.session_state.sharpe_started = False
            st.session_state.sharpe_done = False
            st.session_state.sharpe_results = {}
            st.experimental_rerun()
        return

    if st.button("Run Sharpe analysis", key="run_sharpe"):
        st.session_state.sharpe_started = True

    if not st.session_state.sharpe_started:
        st.info("Press Run Sharpe analysis to calculate ratios and compare strategy versions.")
        return

    input_folder = os.path.join('DATA', 'output_instruments')
    csvs_dictionary = {}
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            inst = file.split('_')[0]
            version = file.split('_')[-1].replace('.csv', '')
            df = pd.read_csv(os.path.join(input_folder, file))
            csvs_dictionary.setdefault(inst, {})[version] = df

    sharpes = []
    for inst, versions in csvs_dictionary.items():
        for version, df in versions.items():
            if version == "results":
                continue
            if 'forecast*returns' in df.columns:
                series = df['forecast*returns'].dropna()
                if not series.empty and series.std() > 0:
                    sharpe = series.mean() / series.std() * np.sqrt(252)
                    sharpes.append({'Instrument': inst, 'Version': version, 'Sharpe Ratio': sharpe})
    sharpes_df = pd.DataFrame(sharpes)

    st.subheader("Sharpe Ratios for forecast*returns")
    st.dataframe(sharpes_df)

    instruments = list(csvs_dictionary.keys())
    selected_inst = st.selectbox("Select Instrument", instruments)

    versions = [v for v in csvs_dictionary[selected_inst].keys() if v != "results"]
    n_versions = len(versions)
    default_weight = 1.0 / n_versions if n_versions > 0 else 0.0

    st.subheader(f"Set Weights for {selected_inst} Versions (sum ≤ 1.0)")
    weights = []
    total_weight = 0.0
    for version in versions:
        weight = st.number_input(
            f"Weight for {version}",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.01,
            key=f"{selected_inst}_{version}_weight"
        )
        weights.append(weight)
        total_weight += weight

    if total_weight > 1.0:
        st.warning("Total weight exceeds 1.0. Weights will be normalized.")
        weights = [w / total_weight for w in weights]
        total_weight = sum(weights)

    st.subheader("Weighted Sharpe Ratios")
    weighted_sharpes = []
    sum_weighted_sharpe = 0.0
    for version, weight in zip(versions, weights):
        sharpe_row = sharpes_df[(sharpes_df['Instrument'] == selected_inst) & (sharpes_df['Version'] == version)]
        if not sharpe_row.empty:
            sharpe = sharpe_row['Sharpe Ratio'].values[0]
            weighted = sharpe * weight
            weighted_sharpes.append({'Version': version, 'Weight': weight, 'Sharpe Ratio': sharpe, 'Weighted Sharpe': weighted})
            sum_weighted_sharpe += weighted

    st.dataframe(pd.DataFrame(weighted_sharpes))
    st.write(f"**Sum of Weighted Sharpe Ratios:** {sum_weighted_sharpe:.4f}")

    st.subheader(f"Returns Time Series for {selected_inst} (first 10 rows per version)")
    tabs = st.tabs(versions)
    for i, version in enumerate(versions):
        df = csvs_dictionary[selected_inst][version]
        with tabs[i]:
            if 'forecast*returns' in df.columns:
                st.write(f"**{selected_inst} - {version}**")
                st.dataframe(df['forecast*returns'].head(10))
                st.line_chart(df['forecast*returns'])

    st.download_button(
        label="Download Sharpe Ratios CSV",
        data=sharpes_df.to_csv(index=False).encode('utf-8'),
        file_name='sharpe_ratios_forecast_returns.csv',
        mime='text/csv'
    )

    sharpe_results_path = os.path.join('DATA', 'output_instruments', 'sharpe_results.json')
    sharpes_dict_to_save = sharpes_df.to_dict(orient='records')
    with open(sharpe_results_path, 'w') as f:
        json.dump(sharpes_dict_to_save, f, indent=2)

    st.session_state.sharpe_results = {
        "sharpes_df": sharpes_df.to_dict(orient="records"),
        "output_path": sharpe_results_path
    }

    returns_list = []
    for version in versions:
        df = csvs_dictionary[selected_inst][version]
        if 'forecast*returns' in df.columns:
            returns_list.append(df['forecast*returns'].reset_index(drop=True))
    if returns_list:
        returns_matrix = pd.concat(returns_list, axis=1).fillna(0)
        weights_arr = np.array(weights)
        weights_input = st.text_input("Enter weights as comma-separated values", value=",".join(map(str, weights_arr)))
        try:
            weights_arr = np.array([float(w) for w in weights_input.split(",")])
        except Exception:
            weights_arr = np.array(weights)

        st.write(f"**Using Weights:** {weights_arr}")
        portfolio_returns = returns_matrix.dot(weights_arr)
        if portfolio_returns.std() > 0:
            portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        else:
            portfolio_sharpe = np.nan
        st.subheader("Overall Portfolio Sharpe Ratio")
        st.write(f"**Portfolio Sharpe Ratio (weighted): {portfolio_sharpe:.4f}**")
        st.line_chart(portfolio_returns, use_container_width=True)
    else:
        st.write("No valid return series found for selected versions.")

    st.session_state.sharpe_done = True
    st.session_state.sharpe_started = False

