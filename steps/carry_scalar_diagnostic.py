"""Diagnostic only - does NOT change any forecast or output.

Pooled Carver-style Carry scalar: the single scalar s such that the average absolute uncapped
forecast, pooled across every instrument-day, equals 10:

    s* = 10 / mean(|raw_carry|)   over all instrument-days with a Carry output

The live Carry strategy still uses a fixed scalar of 30 (strategies/carry.py). This script only
reports what a pooled scalar would be, plus the resulting average capped forecast for reference.
Calibration is in-sample (full history), so it is a diagnostic until back-adjusted history is in.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CARRY_DIR = os.path.join(ROOT, 'DATA', 'output_instruments')
CURRENT_SCALAR = 30.0
TARGET_AVG_ABS = 10.0
CAP = 20.0


def main() -> None:
    frames = []
    for path in sorted(glob.glob(os.path.join(CARRY_DIR, '*_CARRY.csv'))):
        d = pd.read_csv(path)
        if 'raw_carry' not in d.columns:
            continue
        raw = pd.to_numeric(d['raw_carry'], errors='coerce').dropna()
        frames.append(pd.DataFrame({'instrument': os.path.basename(path)[:-len('_CARRY.csv')],
                                    'raw_carry': raw}))

    if not frames:
        raise SystemExit('No *_CARRY.csv outputs with raw_carry found.')

    pooled = pd.concat(frames, ignore_index=True)
    mean_abs_raw = pooled['raw_carry'].abs().mean()
    pooled_scalar = TARGET_AVG_ABS / mean_abs_raw
    avg_capped_at_current = (CURRENT_SCALAR * pooled['raw_carry']).clip(-CAP, CAP).abs().mean()
    avg_capped_at_pooled = (pooled_scalar * pooled['raw_carry']).clip(-CAP, CAP).abs().mean()

    print('Pooled Carry scalar diagnostic (not applied, in-sample)')
    print(f'Instruments: {", ".join(f["instrument"].iloc[0] for f in frames)}')
    print(f'Instrument-days pooled: {len(pooled)}')
    print(f'Mean |raw_carry| pooled: {mean_abs_raw:.6f}')
    print(f'Pooled scalar for avg |forecast| = {TARGET_AVG_ABS:g}: {pooled_scalar:.4f}')
    print(f'Current fixed scalar: {CURRENT_SCALAR:g}')
    print(f'Avg |capped forecast| pooled at current {CURRENT_SCALAR:g}: {avg_capped_at_current:.3f}')
    print(f'Avg |capped forecast| pooled at {pooled_scalar:.4f} (reference): {avg_capped_at_pooled:.3f}')


if __name__ == '__main__':
    main()
