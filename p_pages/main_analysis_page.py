import streamlit as st
import os
import pandas as pd
import numpy as np
from steps.p1_analysis import main_analysis
from steps.app_settings import get_setting, set_settings
from steps.volatility import simple_price_volatility
from steps.pipeline_info import show_active_instruments, show_strategy_explanations, show_source_paths, strategy_source_paths, show_generated_files
import shutil
import stat
import time
import traceback

TRADING_DAYS = 256


def _calc_standard_cost(df: pd.DataFrame, point_value: float) -> float:
    """Standard cost in Sharpe Ratio units, per Carver's Systematic Trading (verified against
    his own Euro Stoxx worked example: SC = 2 x total_one_way_cost / annualised ICV).

    total_one_way_cost = half_bid_ask_spread (converted to $/euro via point_value) +
                          EXECUTION_COST + CLEARING_COST (already in $/euro, no scaling needed)
    annualised ICV = st_dev (calculated in code: std of the last 20 daily price changes, most recent,
    in price points - steps/volatility.py, never the input file's column) x point_value x sqrt(256)

    Falls back to the input file's own supplied 'Standard Cost' column if BID/ASK/
    EXECUTION_COST/CLEARING_COST aren't available for this instrument yet.
    """
    required_cols = {'BID', 'ASK', 'EXECUTION_COST', 'CLEARING_COST'}
    has_cost_inputs = required_cols.issubset(df.columns)
    if has_cost_inputs:
        bid = df['BID'].iloc[0]
        ask = df['ASK'].iloc[0]
        exec_cost = df['EXECUTION_COST'].iloc[0]
        clearing_cost = df['CLEARING_COST'].iloc[0]
        has_cost_inputs = all(pd.notna(v) for v in (bid, ask, exec_cost, clearing_cost))

    if not has_cost_inputs:
        return df['Standard Cost'].iloc[0]

    if 'PX_CLOSE_1D' not in df.columns:
        return df['Standard Cost'].iloc[0]
    st_dev_series = simple_price_volatility(df['PX_CLOSE_1D']).dropna()
    if st_dev_series.empty:
        return df['Standard Cost'].iloc[0]
    st_dev = st_dev_series.iloc[-1]

    half_spread = (ask - bid) / 2.0
    expected_execution_cost = half_spread * point_value
    total_one_way_cost = expected_execution_cost + exec_cost + clearing_cost

    icv_daily = st_dev * point_value
    icv_annual = icv_daily * np.sqrt(TRADING_DAYS)
    if not icv_annual:
        return df['Standard Cost'].iloc[0]

    return 2.0 * total_one_way_cost / icv_annual


def run():
    st.title("Strategy Analysis")
    show_source_paths(["steps/p1_analysis.py :: main_analysis()"])

    show_active_instruments()

    if 'main_analysis_started' not in st.session_state:
        st.session_state.main_analysis_started = False
    if 'main_analysis_done' not in st.session_state:
        st.session_state.main_analysis_done = False
    if 'main_analysis_results' not in st.session_state:
        st.session_state.main_analysis_results = {}

    if st.session_state.main_analysis_done:
        st.success("Strategy Analysis already completed. Use Run Strategy Analysis Again to rerun.")
        results = st.session_state.main_analysis_results
        if results:
            st.subheader("Control sample")
            st.json(results.get("control_sample", {}))
            st.subheader("Input CSV sample")
            for sample in results.get("csv_samples", []):
                st.write(f"Instrument: {sample['instrument']}")
                st.dataframe(sample['head'])
            st.write(results.get("summary", ""))
        show_generated_files(os.path.join('DATA', 'output_instruments'), heading="Generated files (Strategy Analysis)")
        if st.button("Run Strategy Analysis Again", key="rerun_lysis"):
            st.session_state.main_analysis_started = False
            st.session_state.main_analysis_done = False
            st.session_state.main_analysis_results = {}
            st.rerun()
        return

    st.subheader("Strategies to run")
    st.caption(
        "After selecting the strategies below, each will be backtested at a parameter level "
        "(e.g. each EWMA speed, each carry span) with its forecast standardised and capped to "
        "the range +20 / -20."
    )
    run_ewma = st.checkbox("EWMA", value=get_setting('run_ewma', True), key="run_strategy_ewma")
    run_carry = st.checkbox("Carry", value=get_setting('run_carry', False), key="run_strategy_carry")
    run_breakout = st.checkbox("Breakout", value=get_setting('run_breakout', False), key="run_strategy_breakout")
    run_ewma_norm = st.checkbox("EWMA Norm", value=get_setting('run_ewma_norm', False), key="run_strategy_ewma_norm")
    run_carry_spans = st.checkbox(
        "Carry Spans",
        value=get_setting('run_carry_spans', False),
        key="run_strategy_carry_spans",
        help="4-span carry (5/20/60/120 day smoothing), separate from the single-span Carry above.",
    )
    st.checkbox("Vol1", value=False, key="run_strategy_vol1")
    st.checkbox("Vol2", value=False, key="run_strategy_vol2")
    st.checkbox("RV1", value=False, key="run_strategy_rv1")
    st.checkbox("RV2", value=False, key="run_strategy_rv2")
    set_settings({
        'run_ewma': run_ewma,
        'run_carry': run_carry,
        'run_breakout': run_breakout,
        'run_ewma_norm': run_ewma_norm,
        'run_carry_spans': run_carry_spans,
    })

    selected_strategies = []
    if run_ewma:
        selected_strategies.append("EWMA")
    if run_carry:
        selected_strategies.append("CARRY")
    if run_breakout:
        selected_strategies.append("BREAKOUT")
    if run_ewma_norm:
        selected_strategies.append("EWMA_NORM")
    if run_carry_spans:
        selected_strategies.append("CARRY_SPANS")

    if selected_strategies:
        show_source_paths(strategy_source_paths(selected_strategies))
        show_strategy_explanations(selected_strategies)

    if not selected_strategies:
        st.warning("Select at least one strategy before running analysis.")
        return

    if run_breakout:
        st.warning("Breakout is selectable, but this page currently only writes EWMA, Carry, and EWMA Norm output files.")

    if st.button("Run Strategy Analysis", key="run_lysis", type="primary"):
        st.session_state.main_analysis_started = True

    if not st.session_state.main_analysis_started:
        st.info("Press Run Strategy Analysis to start the analysis.")
        return

    st.write(f"Running analysis on all input instruments with: {', '.join(selected_strategies)}")
    input_folder = os.path.join('DATA', 'input_instruments')

    # %%
    ### REMOVE FOLDERS IF THEY EXIST (robust on Windows)
    # helper to handle permission errors when deleting files/folders
    def _on_rm_error(func, path, exc_info):
        """Error handler for shutil.rmtree.
        Attempts to change the file to writable and retries the operation.
        """
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            # if chmod fails, ignore and let the next attempt try to remove
            pass
        try:
            func(path)
        except Exception:
            # as a last resort, if it's a file try os.remove
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                # re-raise the original exception so callers know it failed
                raise

    def safe_rmtree(path, retries=3, delay=0.5):
        """Remove a directory tree with retries and an onerror handler.

        This tries to handle common Windows PermissionError situations by
        making files writable and retrying. If removal ultimately fails,
        a Streamlit warning is shown and the function returns without
        raising an exception (to avoid crashing the app).
        """
        if not os.path.exists(path):
            return

        for attempt in range(1, retries + 1):
            try:
                shutil.rmtree(path, onerror=_on_rm_error)
                return
            except PermissionError as e:
                time.sleep(delay * attempt)
                if attempt == retries:
                    st.warning(f"Could not remove folder {path}: {e}")
                    st.text(traceback.format_exc())
                    return
            except OSError as e:
                time.sleep(delay * attempt)
                if attempt == retries:
                    st.warning(f"Could not remove folder {path}: {e}")
                    st.text(traceback.format_exc())
                    return
            except Exception as e:
                st.warning(f"Unexpected error removing {path}: {e}")
                st.text(traceback.format_exc())
                return

    output_folder = os.path.join('DATA', 'output_instruments')
    if os.path.exists(output_folder):
        safe_rmtree(output_folder)

    combined_forecast_folder = os.path.join('DATA', 'combined_forecast')
    if os.path.exists(combined_forecast_folder):
        safe_rmtree(combined_forecast_folder)

    order_folder = os.path.join('DATA', 'order_folder')
    if os.path.exists(order_folder):
        safe_rmtree(order_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(combined_forecast_folder, exist_ok=True)
    os.makedirs(order_folder, exist_ok=True)

    csv_path = os.path.join('DATA', 'input_main', 'input_main.csv')
    csvs_dictionary = {}

    control_df = pd.read_csv(csv_path)
    control = {}
    for _, row in control_df.iterrows():
        instrument = row['INSTRUMENT']
        control[instrument] = {'INSTRUMENT_WEIGHTS': row['INSTRUMENT_WEIGHTS']}

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_folder, file))

            # Price volatility for ICV (sizing/turnover) and standard cost is calculated in code -
            # the rolling std of the last 20 daily price changes - and replaces any 'st_dev'
            # column in the input file, which is never read. (EWMA's and Carry's own EWMA std used
            # to normalise their forecasts is separate and unchanged.)
            df['st_dev'] = simple_price_volatility(df['PX_CLOSE_1D'])
            
            name = file[:-4]
            csvs_dictionary[name] = df

            if name not in control:
                control[name] = {}

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
                'STANDARD_COST': _calc_standard_cost(df, df['POINT_VALUE'].iloc[0]),
})

    with st.expander("Show control (framework) data sample"):
        st.json({k: control[k] for k in list(control.keys())[:3]})

    with st.expander("Show csvs_dictionary (input CSVs) sample"):
        for k in list(csvs_dictionary.keys())[:3]:
            st.write(f"Instrument: {k}")
            st.dataframe(csvs_dictionary[k].head())

    main_analysis(control, csvs_dictionary, selected_strategies)
    st.success("Analysis complete.")

    output_path = os.path.join(output_folder, 'control_output.csv')
    control_records = []
    for instrument, values in control.items():
        record = {'INSTRUMENT': instrument}
        record.update(values)
        control_records.append(record)

    control_df_output = pd.DataFrame(control_records)
    control_df_output.to_csv(output_path, index=False)
    st.info(f"Control data saved to {output_path}")

    st.session_state.main_analysis_results = {
        "control_sample": {k: control[k] for k in list(control.keys())[:3]},
        "csv_samples": [
            {"instrument": k, "head": csvs_dictionary[k].head().to_dict(orient="records")}
            for k in list(csvs_dictionary.keys())[:3]
        ],
        "summary": f"Processed {len(csvs_dictionary)} instruments with {', '.join(selected_strategies)} and saved control_output.csv."
    }
    st.session_state.main_analysis_done = True
    st.session_state.main_analysis_started = False

    show_generated_files(os.path.join('DATA', 'output_instruments'), heading="Generated files (Strategy Analysis)")

# To load the saved control variable later:
# control_df_loaded = pd.read_csv('DATA/output_instruments/control_output.csv')
# control_loaded = {}
# for _, row in control_df_loaded.iterrows():
#     instrument = row['INSTRUMENT']
#     control_loaded[instrument] = row.drop('INSTRUMENT').to_dict()
