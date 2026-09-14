import os
import shutil
import streamlit as st
import pandas as pd

from steps.checkpoint import create_checkpoint, list_checkpoints, restore_checkpoint
from steps.app_settings import load_settings, save_settings

INPUT_MAIN_CSV = os.path.join('DATA', 'input_main', 'input_main.csv')
INPUT_INSTRUMENTS_DIR = os.path.join('DATA', 'input_instruments')
ALL_INPUT_FILES_DIR = os.path.join('DATA', 'all_input_files')

REQUIRED_COLUMNS = [
    'Date', 'PX_CLOSE_1D', 'CRNCY', 'EXCHANGE', 'SECTYPE', 'TICK_SIZE',
    'TICK_VALUE', 'POINT_VALUE', 'CONTRACT_VALUE', 'Exchange rate', 'Standard Cost',
]


def _active_instruments() -> list:
    if not os.path.isdir(INPUT_INSTRUMENTS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(INPUT_INSTRUMENTS_DIR) if f.endswith('.csv'))


def _scan_pool() -> pd.DataFrame:
    active = set(_active_instruments())
    rows = []
    if os.path.isdir(ALL_INPUT_FILES_DIR):
        for f in sorted(os.listdir(ALL_INPUT_FILES_DIR)):
            if not f.endswith('.csv'):
                continue
            name = f[:-4]
            try:
                cols = set(pd.read_csv(os.path.join(ALL_INPUT_FILES_DIR, f), nrows=0).columns)
            except Exception:
                cols = set()
            missing = [c for c in REQUIRED_COLUMNS if c not in cols]
            rows.append({
                'Active': name in active,
                'Instrument': name,
                'Compatible': len(missing) == 0,
                'Missing columns': ', '.join(missing),
            })

    pool_names = {r['Instrument'] for r in rows}
    for name in sorted(active - pool_names):
        rows.append({
            'Active': True,
            'Instrument': name,
            'Compatible': True,
            'Missing columns': '(not in all_input_files/ pool)',
        })

    if not rows:
        return pd.DataFrame(columns=['Active', 'Instrument', 'Compatible', 'Missing columns'])
    return pd.DataFrame(rows).sort_values('Instrument').reset_index(drop=True)


def _load_weights_map() -> dict:
    if not os.path.exists(INPUT_MAIN_CSV):
        return {}
    df = pd.read_csv(INPUT_MAIN_CSV)
    if 'INSTRUMENT' not in df.columns or 'INSTRUMENT_WEIGHTS' not in df.columns:
        return {}
    return dict(zip(df['INSTRUMENT'], df['INSTRUMENT_WEIGHTS']))


def _build_weights_df() -> pd.DataFrame:
    active = _active_instruments()
    weights_map = _load_weights_map()
    rows = [{'INSTRUMENT': inst, 'INSTRUMENT_WEIGHTS': weights_map.get(inst)} for inst in active]
    return pd.DataFrame(rows, columns=['INSTRUMENT', 'INSTRUMENT_WEIGHTS'])


def _orphan_weights() -> list:
    active = set(_active_instruments())
    weights_map = _load_weights_map()
    return sorted(set(weights_map.keys()) - active)


def run():
    st.title("Settings")
    st.caption("Edit portfolio configuration here instead of hand-editing CSV files.")

    # ---------------- Instrument Pool ----------------
    st.subheader("Instrument Pool")
    st.caption(
        f"Choose which instruments from {ALL_INPUT_FILES_DIR} are active. Active instruments are "
        f"copied into {INPUT_INSTRUMENTS_DIR} and used by every pipeline step. Everything currently "
        "active is checked by default - uncheck any to run a customized subset."
    )

    if 'settings_pool_df' not in st.session_state:
        st.session_state.settings_pool_df = _scan_pool()

    pool_edited = st.data_editor(
        st.session_state.settings_pool_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            'Active': st.column_config.CheckboxColumn('Active'),
            'Instrument': st.column_config.TextColumn('Instrument', disabled=True),
            'Compatible': st.column_config.CheckboxColumn('Compatible', disabled=True),
            'Missing columns': st.column_config.TextColumn('Missing columns', disabled=True),
        },
        key="settings_pool_editor",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reload instrument pool from disk", key="reload_pool"):
            st.session_state.settings_pool_df = _scan_pool()
            st.rerun()
    with col_b:
        apply_pool = st.button("Apply instrument selection", key="apply_pool", type="primary")

    if apply_pool:
        blocked, added, removed = [], [], []
        checkpoint_path = create_checkpoint(reason='pre-instrument-selection')
        os.makedirs(INPUT_INSTRUMENTS_DIR, exist_ok=True)

        for _, row in pool_edited.iterrows():
            name = row['Instrument']
            want_active = bool(row['Active'])
            src = os.path.join(ALL_INPUT_FILES_DIR, f'{name}.csv')
            dest = os.path.join(INPUT_INSTRUMENTS_DIR, f'{name}.csv')

            if want_active and not bool(row['Compatible']) and os.path.exists(src):
                blocked.append(name)
                continue

            if want_active:
                if not os.path.exists(dest) and os.path.exists(src):
                    shutil.copy2(src, dest)
                    added.append(name)
            else:
                if os.path.exists(dest):
                    os.remove(dest)
                    removed.append(name)

        st.session_state.settings_pool_df = _scan_pool()
        st.session_state.pop('settings_weights_df', None)

        if blocked:
            st.warning("Not activated - incompatible columns: " + ", ".join(blocked))
        st.success(
            f"Instrument selection applied (checkpoint: {checkpoint_path}). "
            f"Added: {', '.join(added) if added else 'none'}. "
            f"Removed: {', '.join(removed) if removed else 'none'}."
        )
        st.rerun()

    st.divider()

    # ---------------- Instrument Weights ----------------
    st.subheader("Instrument Weights")
    st.caption(f"Weights for the active instruments above. Saved to {INPUT_MAIN_CSV}.")

    if 'settings_weights_df' not in st.session_state:
        st.session_state.settings_weights_df = _build_weights_df()

    weights_df = st.session_state.settings_weights_df

    if weights_df.empty:
        st.info("No active instruments yet - activate some in the Instrument Pool above.")
    else:
        missing_weight = weights_df[weights_df['INSTRUMENT_WEIGHTS'].isna()]['INSTRUMENT'].tolist()
        if missing_weight:
            st.warning("No weight set yet for: " + ", ".join(missing_weight))

    orphans = _orphan_weights()
    if orphans:
        label = f"{len(orphans)} unused weight entr{'y' if len(orphans) == 1 else 'ies'} not linked to an active instrument"
        with st.expander(label):
            st.write(", ".join(orphans))
            st.caption(
                "These exist in input_main.csv but have no matching active instrument, so they're "
                "hidden above and will be dropped next time you save weights."
            )

    edited_weights = st.data_editor(
        weights_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "INSTRUMENT": st.column_config.TextColumn("Instrument", disabled=True),
            "INSTRUMENT_WEIGHTS": st.column_config.NumberColumn("Weight", step=0.01, format="%.4f"),
        },
        key="settings_weights_editor",
    )
    st.session_state.settings_weights_df = edited_weights

    weight_sum = pd.to_numeric(edited_weights['INSTRUMENT_WEIGHTS'], errors='coerce').sum()
    st.metric("Sum of weights", f"{weight_sum:.4f}")
    if abs(weight_sum - 1.0) > 1e-6:
        st.warning("Weights don't sum to 1.0 - double check before saving if that's not intentional.")

    if st.button("Save weights", key="save_weights", type="primary"):
        save_df = edited_weights[['INSTRUMENT', 'INSTRUMENT_WEIGHTS']].dropna(subset=['INSTRUMENT'])
        checkpoint_path = create_checkpoint(reason='pre-settings-save')
        os.makedirs(os.path.dirname(INPUT_MAIN_CSV), exist_ok=True)
        save_df.to_csv(INPUT_MAIN_CSV, index=False)
        st.session_state.settings_weights_df = _build_weights_df()
        st.success(
            f"Saved weights to {INPUT_MAIN_CSV} (checkpoint saved to {checkpoint_path}). "
            "Re-run Main Analysis so control_output.csv picks up the new weights."
        )

    st.divider()
    st.subheader("Run Options")
    st.caption(
        "Choices made on other tabs (which strategies to run, validation sample size, etc.) are saved "
        "here automatically so you don't have to re-pick them next time."
    )
    persisted = load_settings()
    if not persisted:
        st.info("Nothing saved yet - options you pick on other tabs will show up here.")
    else:
        st.json(persisted)
        if st.button("Reset run options to defaults", key="reset_run_options"):
            save_settings({})
            st.success("Run options cleared - other tabs will fall back to their defaults.")
            st.rerun()

    st.divider()
    st.subheader("Checkpoints")
    st.caption("Snapshots of input_main.csv, input_instruments/, and control_output.csv.")

    if st.button("Create checkpoint now", key="manual_checkpoint"):
        path = create_checkpoint(reason='manual')
        st.success(f"Checkpoint created at {path}")

    checkpoints = list_checkpoints()
    if not checkpoints:
        st.info("No checkpoints yet.")
        return

    def _label(c):
        when = c['created'].strftime('%Y-%m-%d %H:%M:%S') if c['created'] else c['name']
        return f"{when} ({c['reason']})"

    options = {_label(c): c['path'] for c in checkpoints}
    selected_label = st.selectbox("Select a checkpoint to restore", list(options.keys()), key="checkpoint_select")

    st.warning(
        "Restoring will overwrite the current input_main.csv, input_instruments/ files, "
        "and control_output.csv. A pre-restore checkpoint is taken automatically first."
    )
    if st.button("Restore selected checkpoint", key="restore_checkpoint_btn"):
        restore_checkpoint(options[selected_label])
        st.session_state.pop('settings_weights_df', None)
        st.session_state.pop('settings_pool_df', None)
        st.success(f"Restored checkpoint: {selected_label}")
        st.rerun()

    with st.expander("All checkpoints"):
        st.dataframe(pd.DataFrame([
            {'Checkpoint': c['name'], 'Created': c['created'], 'Reason': c['reason']}
            for c in checkpoints
        ]))
