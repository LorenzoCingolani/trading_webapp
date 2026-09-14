import os
import shutil
from datetime import datetime
from typing import Dict, List

CHECKPOINT_ROOT = os.path.join('DATA', 'checkpoints')

INPUT_MAIN_CSV = os.path.join('DATA', 'input_main', 'input_main.csv')
INPUT_INSTRUMENTS_DIR = os.path.join('DATA', 'input_instruments')
CONTROL_OUTPUT_CSV = os.path.join('DATA', 'output_instruments', 'control_output.csv')


def create_checkpoint(reason: str = 'manual') -> str:
    """
    Snapshot the current config files (instrument weights, instrument CSVs,
    control output) into a new timestamped folder under DATA/checkpoints/.

    Returns the path to the created checkpoint folder.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f'{timestamp}_{reason}')

    if os.path.exists(INPUT_MAIN_CSV):
        dest = os.path.join(checkpoint_dir, 'input_main')
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(INPUT_MAIN_CSV, os.path.join(dest, 'input_main.csv'))

    if os.path.isdir(INPUT_INSTRUMENTS_DIR):
        instrument_files = [f for f in os.listdir(INPUT_INSTRUMENTS_DIR) if f.endswith('.csv')]
        if instrument_files:
            dest = os.path.join(checkpoint_dir, 'input_instruments')
            os.makedirs(dest, exist_ok=True)
            for f in instrument_files:
                shutil.copy2(os.path.join(INPUT_INSTRUMENTS_DIR, f), os.path.join(dest, f))

    if os.path.exists(CONTROL_OUTPUT_CSV):
        dest = os.path.join(checkpoint_dir, 'output_instruments')
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(CONTROL_OUTPUT_CSV, os.path.join(dest, 'control_output.csv'))

    return checkpoint_dir


def list_checkpoints() -> List[Dict]:
    """
    List existing checkpoints under DATA/checkpoints/, newest first.
    """
    if not os.path.isdir(CHECKPOINT_ROOT):
        return []

    checkpoints = []
    for name in os.listdir(CHECKPOINT_ROOT):
        path = os.path.join(CHECKPOINT_ROOT, name)
        if not os.path.isdir(path):
            continue
        timestamp_part, _, reason = name.partition('_')
        try:
            created = datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')
        except ValueError:
            created = None
        checkpoints.append({
            'name': name,
            'path': path,
            'created': created,
            'reason': reason or name,
        })

    checkpoints.sort(key=lambda c: c['created'] or datetime.min, reverse=True)
    return checkpoints


def restore_checkpoint(checkpoint_dir: str) -> None:
    """
    Restore config files from a checkpoint folder back to their original
    locations, overwriting current versions. Files present now but not in
    the checkpoint are left untouched. Takes a fresh 'pre-restore' checkpoint
    of the current state first, so the restore itself can be undone.
    """
    create_checkpoint(reason='pre-restore')

    src_input_main = os.path.join(checkpoint_dir, 'input_main', 'input_main.csv')
    if os.path.exists(src_input_main):
        os.makedirs(os.path.dirname(INPUT_MAIN_CSV), exist_ok=True)
        shutil.copy2(src_input_main, INPUT_MAIN_CSV)

    src_instruments_dir = os.path.join(checkpoint_dir, 'input_instruments')
    if os.path.isdir(src_instruments_dir):
        os.makedirs(INPUT_INSTRUMENTS_DIR, exist_ok=True)
        for f in os.listdir(src_instruments_dir):
            shutil.copy2(os.path.join(src_instruments_dir, f), os.path.join(INPUT_INSTRUMENTS_DIR, f))

    src_control_output = os.path.join(checkpoint_dir, 'output_instruments', 'control_output.csv')
    if os.path.exists(src_control_output):
        os.makedirs(os.path.dirname(CONTROL_OUTPUT_CSV), exist_ok=True)
        shutil.copy2(src_control_output, CONTROL_OUTPUT_CSV)
