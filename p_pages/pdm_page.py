import streamlit as st
import os
import pandas as pd
from steps.p3_pdm import pdm_main, PDM_UPPER_BOUND
from steps.pipeline_info import show_active_instruments, show_step_explanation, show_source_paths, show_generated_files

PDM_OUTPUT_FILE = os.path.join('DATA', 'combinedForecast', 'PDM_portfolio.h5')

INPUT_MAIN_CSV = os.path.join('DATA', 'input_main', 'input_main.csv')

def run():
    st.title("PDM")
    show_source_paths(["steps/p3_pdm.py :: pdm_main()"])
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
        show_generated_files([PDM_OUTPUT_FILE], heading="Generated files (PDM)")
        if st.button("Run PDM again", key="rerun_pdm"):
            st.session_state.pdm_started = False
            st.session_state.pdm_done = False
            st.session_state.pdm_results = {}
            st.rerun()
        return

    if st.button("Run PDM", key="run_pdm", type="primary"):
        st.session_state.pdm_started = True

    if not st.session_state.pdm_started:
        st.info("Press Run PDM to calculate the portfolio diversification multiplier.")
        return

    if not os.path.exists(INPUT_MAIN_CSV):
        st.error('No instrument weights found. Set weights for your active instruments on the Settings tab first.')
        return

    weights_df = pd.read_csv(INPUT_MAIN_CSV)
    weights_map = dict(zip(weights_df.get('INSTRUMENT', []), weights_df.get('INSTRUMENT_WEIGHTS', [])))

    csvs_dictionary = {}
    control = {}
    missing_weights = []

    for file in os.listdir(input_folder):
        if not file.endswith('.csv'):
            continue
        instrument_name = file[:-4]
        weight = weights_map.get(instrument_name)
        if weight is None or pd.isna(weight):
            missing_weights.append(instrument_name)
            continue
        csvs_dictionary[instrument_name] = pd.read_csv(os.path.join(input_folder, file))
        control[instrument_name] = {'INSTRUMENT_WEIGHTS': weight}

    if missing_weights:
        st.warning(
            "Skipping instruments with no weight set: " + ", ".join(missing_weights) +
            ". Set their weights on the Settings tab to include them."
        )

    if not csvs_dictionary:
        st.error(
            "No active instrument has a weight set, so PDM can't be calculated. "
            "Go to the Settings tab to activate instruments and set their weights."
        )
        return

    pdm_result = pdm_main(control, csvs_dictionary)
    st.success("PDM process complete.")
    st.write("PDM result:", pdm_result)
    st.metric("Portfolio Diversification Multiplier (PDM)", f"{pdm_result:.4f}")

    st.session_state.pdm_results = {"pdm_result": pdm_result}
    st.session_state.pdm_done = True
    st.session_state.pdm_started = False

    show_generated_files([PDM_OUTPUT_FILE], heading="Generated files (PDM)")
