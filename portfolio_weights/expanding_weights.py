"""Expanding-window Carver handcrafting weights, rebalanced annually.

At each calendar year-end rebalance date R, the handcrafting function (portfolio_weights.py,
used unchanged) is run on the raw weekly subsystem returns up to and including R only, after
standardising each column to unit volatility over that same window. The resulting weights apply
from the following week onwards, until the next rebalance.

Rebalances where the handcrafting step can't run yet (e.g. not enough pairwise overlap history)
are logged with status and produce no weights; weeks before the first successful rebalance have
no weights. The handcrafting IDM is deliberately not saved or passed downstream.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
RAW_PATH = os.path.join(ROOT, 'DATA', 'portfolio_weights', 'weekly_subsystem_returns_raw.csv')
APPLIED_PATH = os.path.join(ROOT, 'DATA', 'portfolio_weights', 'expanding_weights_applied.csv')
REBALANCE_PATH = os.path.join(ROOT, 'DATA', 'portfolio_weights', 'expanding_weights_rebalances.csv')
CURRENT_PATH = os.path.join(ROOT, 'DATA', 'portfolio_weights', 'current_weights.csv')
MIN_WINDOW_WEEKS = 104
OVERLAP_ERROR_TEXT = 'overlapping weeks'


def load_handcrafting():
    spec = importlib.util.spec_from_file_location(
        'handcrafting', os.path.join(HERE, 'portfolio_weights.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def year_end_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.year).max())


def window_weights(raw: pd.DataFrame, rebalance_date: pd.Timestamp, handcrafting) -> pd.Series:
    window = raw.loc[:rebalance_date]
    vol = window.std()
    bad = vol[~np.isfinite(vol) | (vol <= 0)]
    if len(bad):
        raise ValueError(f"Invalid subsystem volatility: {bad.index.tolist()}")
    standardised = window.div(vol, axis=1)
    return handcrafting.handcrafted_instrument_weights(standardised)['weights']


def main() -> None:
    raw = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True).sort_index()
    handcrafting = load_handcrafting()
    weeks = raw.index

    log_rows = []
    for rebalance_date in year_end_rebalance_dates(weeks):
        window_len = int(len(raw.loc[:rebalance_date]))
        later = weeks[weeks > rebalance_date]
        apply_from = later[0] if len(later) else pd.NaT
        w = pd.Series(np.nan, index=raw.columns)
        if window_len < MIN_WINDOW_WEEKS:
            status = f'skipped: {window_len} weeks < MIN_WINDOW_WEEKS={MIN_WINDOW_WEEKS}'
        else:
            try:
                w = window_weights(raw, rebalance_date, handcrafting).reindex(raw.columns)
                status = 'ok'
            except ValueError as exc:
                if OVERLAP_ERROR_TEXT not in str(exc):
                    raise
                status = f'skipped: {exc}'
        log_rows.append({'rebalance_date': rebalance_date, 'apply_from': apply_from,
                         'window_weeks': window_len, 'status': status, **w.to_dict()})

    log = pd.DataFrame(log_rows)
    instruments = list(raw.columns)
    ok = log['status'] == 'ok'
    prev = None
    turnover = []
    for _, row in log.iterrows():
        if row['status'] != 'ok':
            turnover.append(np.nan)
            continue
        if prev is None:
            turnover.append(np.nan)
        else:
            turnover.append(0.5 * float(np.abs(row[instruments].astype(float) - prev).sum()))
        prev = row[instruments].astype(float)
    log['turnover_vs_prev'] = turnover

    usable = log[ok & log['apply_from'].notna()].copy()

    applied = pd.DataFrame(np.nan, index=weeks, columns=raw.columns)
    if not usable.empty:
        from_weights = usable.set_index('apply_from')[list(raw.columns)]
        applied = from_weights.reindex(weeks, method='ffill')
    applied.index.name = 'Date'

    os.makedirs(os.path.dirname(APPLIED_PATH), exist_ok=True)
    applied.to_csv(APPLIED_PATH, date_format='%Y-%m-%d')
    log.to_csv(REBALANCE_PATH, index=False, date_format='%Y-%m-%d')

    if ok.any():
        latest = log[ok].iloc[-1]
        current = pd.DataFrame([{'as_of': latest['rebalance_date'], 'window_weeks': latest['window_weeks'],
                                 **{i: latest[i] for i in instruments}}])
        current.to_csv(CURRENT_PATH, index=False, date_format='%Y-%m-%d')
        print(f'Current weights (as of {latest["rebalance_date"].date()}): {CURRENT_PATH}')
    else:
        print('No successful rebalance yet, so no current weights written.')

    print(log[['rebalance_date', 'window_weeks', 'status', 'turnover_vs_prev']].to_string(index=False))
    print(f'Applied weights: {APPLIED_PATH}')
    print(f'Rebalance log:   {REBALANCE_PATH}')


if __name__ == '__main__':
    main()
