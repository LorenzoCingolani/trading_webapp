import streamlit as st
import os
import pandas as pd
from steps.p3_pdm import pdm_main, PDM_UPPER_BOUND
from steps.pipeline_info import show_active_instruments, show_step_explanation

def run():
    st.title("PDM")
    input_folder = os.path.join('DATA', 'input_instruments')

    show_active_instruments()
    show_step_explanation(
        "Portfolio Diversification Multiplier = `1 / sqrt(wᵀCw)`, where `w` are the instrument weights "
        "and `C` is the correlation matrix of daily % price changes across the active instruments above. "
        f"Rewards a less-correlated portfolio with a bigger position-sizing multiplier, capped at "
        f"{PDM_UPPER_BOUND:.1f}. If any active instrument is missing a weight, the result is NaN - "
        "set weights for all active instruments on the Settings tab first."
    )

    if 'pdm_started' not in st.session_state:
        st.session_state.pdm_started = False
    if 'pdm_done' not in st.session_state:
        st.session_state.pdm_done = False
    if 'pdm_results' not in st.session_state:
        st.session_state.pdm_results = {}

    if st.session_state.pdm_done:
        st.success("PDM already calculated. Use Run PDM again to rerun.")
        results = st.session_state.pdm_results
        if results:
            st.write("PDM result:", results.get("pdm_result"))
            st.metric("Portfolio Diversification Multiplier (PDM)", f"{results.get('pdm_result', 0):.4f}")
        if st.button("Run PDM again", key="rerun_pdm"):
            st.session_state.pdm_started = False
            st.session_state.pdm_done = False
            st.session_state.pdm_results = {}
            st.rerun()
        return

    if st.button("Run PDM", key="run_pdm"):
        st.session_state.pdm_started = True

    if not st.session_state.pdm_started:
        st.info("Press Run PDM to calculate the portfolio diversification multiplier.")
        return

    csv_path = os.path.join('DATA', 'output_instruments', 'control_output.csv')
    csvs_dictionary = {}

    if not os.path.exists(csv_path):
        st.error('No control output found. Expected: DATA/output_instruments/control_output.csv. Run Main Analysis first.')
        return

    control_df = pd.read_csv(csv_path)
    control = {}
    for _, row in control_df.iterrows():
        instrument = row['INSTRUMENT']
        values = row.drop(labels=['INSTRUMENT']).to_dict()
        values['INSTRUMENT'] = instrument
        control[instrument] = values

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            instrument_name = file[:-4]
            if instrument_name in control:
                df = pd.read_csv(os.path.join(input_folder, file))
                csvs_dictionary[instrument_name] = df

    pdm_result = pdm_main(control, csvs_dictionary)
    st.success("PDM process complete.")
    st.write("PDM result:", pdm_result)
    st.metric("Portfolio Diversification Multiplier (PDM)", f"{pdm_result:.4f}")

    st.session_state.pdm_results = {"pdm_result": pdm_result}
    st.session_state.pdm_done = True
    st.session_state.pdm_started = False
