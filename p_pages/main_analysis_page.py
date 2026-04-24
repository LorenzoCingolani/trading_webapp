import streamlit as st
import os
import pandas as pd
from steps.p1_analysis import main_analysis
import shutil
import stat
import time
import traceback

def run():
    st.title("lysis")

    if 'main_analysis_started' not in st.session_state:
        st.session_state.main_analysis_started = False
    if 'main_analysis_done' not in st.session_state:
        st.session_state.main_analysis_done = False
    if 'main_analysis_results' not in st.session_state:
        st.session_state.main_analysis_results = {}

    if st.session_state.main_analysis_done:
        st.success("Analysis already completed. Use Run lysis again to rerun.")
        results = st.session_state.main_analysis_results
        if results:
            st.subheader("Control sample")
            st.json(results.get("control_sample", {}))
            st.subheader("Input CSV sample")
            for sample in results.get("csv_samples", []):
                st.write(f"Instrument: {sample['instrument']}")
                st.dataframe(sample['head'])
            st.write(results.get("summary", ""))
        if st.button("Run lysis again", key="rerun_lysis"):
            st.session_state.main_analysis_started = False
            st.session_state.main_analysis_done = False
            st.session_state.main_analysis_results = {}
            st.experimental_rerun()
        return

    if st.button("Run lysis", key="run_lysis"):
        st.session_state.main_analysis_started = True

    if not st.session_state.main_analysis_started:
        st.info("Press Run lysis to start the analysis.")
        return

    st.write("Running lysis on all input instruments...")
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

    with st.expander("Show control (framework) data sample"):
        st.json({k: control[k] for k in list(control.keys())[:3]})

    with st.expander("Show csvs_dictionary (input CSVs) sample"):
        for k in list(csvs_dictionary.keys())[:3]:
            st.write(f"Instrument: {k}")
            st.dataframe(csvs_dictionary[k].head())

    main_analysis(control, csvs_dictionary)
    st.success("lysis complete.")

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
        "summary": f"Processed {len(csvs_dictionary)} instruments and saved control_output.csv."
    }
    st.session_state.main_analysis_done = True
    st.session_state.main_analysis_started = False

# To load the saved control variable later:
# control_df_loaded = pd.read_csv('DATA/output_instruments/control_output.csv')
# control_loaded = {}
# for _, row in control_df_loaded.iterrows():
#     instrument = row['INSTRUMENT']
#     control_loaded[instrument] = row.drop('INSTRUMENT').to_dict()