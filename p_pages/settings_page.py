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


STATIC_NUMERIC_COLUMNS = ['TICK_SIZE', 'TICK_VALUE', 'POINT_VALUE', 'CONTRACT_VALUE', 'Exchange rate', 'Standard Cost']


def _is_file_locked(path: str) -> bool:
    """
    Renaming a file to itself needs an exclusive handle on Windows, so it fails if another
    program (Excel, a text editor, etc.) has the file open - a cheap, reliable "is this open
    elsewhere" check that doesn't require reading the file.
    """
    if not os.path.exists(path):
        return False
    try:
        os.rename(path, path)
        return False
    except OSError:
        return True


def _scan_pool() -> pd.DataFrame:
    active = set(_active_instruments())
    rows = []
    if os.path.isdir(ALL_INPUT_FILES_DIR):
        for f in sorted(os.listdir(ALL_INPUT_FILES_DIR)):
            if not f.endswith('.csv'):
                continue
            name = f[:-4]
            src = os.path.join(ALL_INPUT_FILES_DIR, f)
            dest = os.path.join(INPUT_INSTRUMENTS_DIR, f)
            try:
                cols = set(pd.read_csv(src, nrows=0).columns)
            except Exception:
                cols = set()
            missing = [c for c in REQUIRED_COLUMNS if c not in cols]

            locked = _is_file_locked(src) or (name in active and _is_file_locked(dest))

            rows.append({
                'Active': name in active,
                'Instrument': name,
                'Compatible': len(missing) == 0,
                'Missing columns': ', '.join(missing),
                'Locked': locked,
                'Data OK': None,
                'Data Issues': 'Not checked yet - click "Validate data" below',
            })

    pool_names = {r['Instrument'] for r in rows}
    for name in sorted(active - pool_names):
        dest = os.path.join(INPUT_INSTRUMENTS_DIR, f'{name}.csv')
        rows.append({
            'Active': True,
            'Instrument': name,
            'Compatible': True,
            'Missing columns': '(not in all_input_files/ pool)',
            'Locked': _is_file_locked(dest),
            'Data OK': None,
            'Data Issues': 'Not checked yet - click "Validate data" below',
        })

    cols = ['Active', 'Instrument', 'Compatible', 'Missing columns', 'Locked', 'Data OK', 'Data Issues']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values('Instrument').reset_index(drop=True)


def _deep_validate_file(path: str) -> dict:
    """
    Mirrors the parsing the pipeline actually does, so a failure here matches a real
    crash later: Date via steps/p3_pdm.py's two-step parse (exact format, then mixed
    dayfirst), PX_CLOSE_1D and the per-instrument static fields (TICK_SIZE etc, read via
    .iloc[0] in main_analysis_page.py) as numeric.
    """
    try:
        df = pd.read_csv(path)
    except Exception as ex:
        return {'ok': False, 'issues': f'could not read file: {ex}'}

    issues = []

    if 'Date' not in df.columns:
        issues.append('no Date column')
    else:
        raw_dates = df['Date']
        try:
            pd.to_datetime(raw_dates, format='%d/%m/%Y', errors='raise')
        except Exception:
            try:
                pd.to_datetime(raw_dates, format='mixed', dayfirst=True, errors='raise')
            except Exception:
                coerced = pd.to_datetime(raw_dates, format='mixed', dayfirst=True, errors='coerce')
                bad_mask = coerced.isna() & raw_dates.notna() & (raw_dates.astype(str).str.strip() != '')
                bad_rows = df.index[bad_mask]
                if len(bad_rows):
                    first = bad_rows[0]
                    issues.append(
                        f"{len(bad_rows)} unparseable date(s) - e.g. row {first}: '{raw_dates.loc[first]}'"
                    )
                else:
                    issues.append('Date column failed to parse')

    if 'PX_CLOSE_1D' in df.columns:
        px = df['PX_CLOSE_1D']
        bad_px = pd.to_numeric(px, errors='coerce').isna() & px.notna()
        if bad_px.any():
            issues.append(f"{int(bad_px.sum())} non-numeric PX_CLOSE_1D value(s)")

    if len(df):
        for col in STATIC_NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            first_val = df[col].iloc[0]
            if pd.isna(pd.to_numeric(pd.Series([first_val]), errors='coerce')).iloc[0]:
                issues.append(f"{col} blank/non-numeric in first row")

    return {'ok': len(issues) == 0, 'issues': '; '.join(issues) if issues else 'OK'}


def _rescan_pool_preserving_checks(previous: pd.DataFrame) -> pd.DataFrame:
    fresh = _scan_pool()
    if previous is None or previous.empty:
        return fresh
    prev_by_name = previous.set_index('Instrument')[['Data OK', 'Data Issues']]
    for idx, row in fresh.iterrows():
        if row['Instrument'] in prev_by_name.index:
            fresh.loc[idx, 'Data OK'] = prev_by_name.loc[row['Instrument'], 'Data OK']
            fresh.loc[idx, 'Data Issues'] = prev_by_name.loc[row['Instrument'], 'Data Issues']
    return fresh


def _load_weights_map() -> dict:
    if not os.path.exists(INPUT_MAIN_CSV):
        return {}
    df = pd.read_csv(INPUT_MAIN_CSV)
    if 'INSTRUMENT' not in df.columns or 'INSTRUMENT_WEIGHTS' not in df.columns:
        return {}
    return dict(zip(df['INSTRUMENT'], df['INSTRUMENT_WEIGHTS']))


def _equal_weights_df(active: list) -> pd.DataFrame:
    share = 1.0 / len(active) if active else 0.0
    rows = [{'INSTRUMENT': inst, 'INSTRUMENT_WEIGHTS': share} for inst in active]
    return pd.DataFrame(rows, columns=['INSTRUMENT', 'INSTRUMENT_WEIGHTS'])


def _build_weights_df() -> pd.DataFrame:
    active = _active_instruments()
    weights_map = _load_weights_map()

    any_missing = any(pd.isna(weights_map.get(inst)) for inst in active)
    if any_missing and active:
        # New/unweighted instruments default to an equal split across everything active -
        # edit and Save to customize; once every active instrument has an explicit weight,
        # this default no longer kicks in.
        return _equal_weights_df(active)

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
            'Locked': st.column_config.CheckboxColumn(
                'Locked', disabled=True,
                help="File is currently open in another program (e.g. Excel) - reading/activating "
                     "it may fail or use stale data until it's closed.",
            ),
            'Data OK': st.column_config.CheckboxColumn('Data OK', disabled=True),
            'Data Issues': st.column_config.TextColumn('Data Issues', disabled=True, width='large'),
        },
        key="settings_pool_editor",
    )

    locked_now = st.session_state.settings_pool_df[st.session_state.settings_pool_df['Locked'] == True]
    if not locked_now.empty:
        st.warning(
            "Currently open in another program (close it before activating/re-validating): "
            + ", ".join(locked_now['Instrument'])
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Reload Instrument Pool from Disk", key="reload_pool"):
            st.session_state.settings_pool_df = _scan_pool()
            st.rerun()
    with col_b:
        validate_data = st.button(
            "Validate Data (Dates & Values)", key="validate_pool_data",
            help="Actually reads every file and checks Date parses cleanly and PX_CLOSE_1D / "
                 "TICK_SIZE / TICK_VALUE / POINT_VALUE / CONTRACT_VALUE / Exchange rate / "
                 "Standard Cost are numeric - catches the errors that otherwise only show up "
                 "later when a pipeline step crashes.",
        )
    with col_c:
        apply_pool = st.button("Apply Instrument Selection", key="apply_pool", type="primary")

    if validate_data:
        current = st.session_state.settings_pool_df.copy()
        with st.spinner(f"Reading and validating {len(current)} file(s)..."):
            for idx, row in current.iterrows():
                src = os.path.join(ALL_INPUT_FILES_DIR, f"{row['Instrument']}.csv")
                if not os.path.exists(src):
                    current.loc[idx, 'Data OK'] = None
                    current.loc[idx, 'Data Issues'] = '(file not found in pool)'
                    continue
                result = _deep_validate_file(src)
                current.loc[idx, 'Data OK'] = result['ok']
                current.loc[idx, 'Data Issues'] = result['issues']
        st.session_state.settings_pool_df = current
        st.rerun()

    if apply_pool:
        blocked, blocked_data, blocked_locked, failed, added, removed = [], [], [], [], [], []
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

            if want_active and bool(row.get('Locked')):
                blocked_locked.append(name)
                continue

            data_ok_val = row.get('Data OK')
            data_checked_and_bad = (
                data_ok_val is not None and not pd.isna(data_ok_val) and not bool(data_ok_val)
            )
            if want_active and data_checked_and_bad:
                blocked_data.append(name)
                continue

            try:
                if want_active:
                    if not os.path.exists(dest) and os.path.exists(src):
                        shutil.copy2(src, dest)
                        added.append(name)
                else:
                    if os.path.exists(dest):
                        os.remove(dest)
                        removed.append(name)
            except OSError as ex:
                failed.append(f"{name} ({ex})")

        st.session_state.settings_pool_df = _rescan_pool_preserving_checks(st.session_state.settings_pool_df)
        st.session_state.pop('settings_weights_df', None)

        if blocked:
            st.warning("Not activated - incompatible columns: " + ", ".join(blocked))
        if blocked_data:
            st.warning("Not activated - failed data validation: " + ", ".join(blocked_data))
        if blocked_locked:
            st.warning("Not activated - file currently open in another program: " + ", ".join(blocked_locked))
        if failed:
            st.error("Failed to update (close the file and try again): " + "; ".join(failed))
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
        weights_map = _load_weights_map()
        if any(pd.isna(weights_map.get(inst)) for inst in weights_df['INSTRUMENT']):
            st.info(
                "New/unweighted instruments default to an equal split across all active instruments. "
                "Edit any value below and click Save to customize."
            )

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

    col_eq, col_save = st.columns(2)
    with col_eq:
        if not weights_df.empty and st.button("Equalize Weights", key="equalize_weights"):
            st.session_state.settings_weights_df = _equal_weights_df(list(weights_df['INSTRUMENT']))
            st.rerun()
    with col_save:
        save_clicked = st.button("Save Weights", key="save_weights", type="primary")

    if save_clicked:
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
