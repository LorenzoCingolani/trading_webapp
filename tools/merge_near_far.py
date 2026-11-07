"""Merge Bloomberg 1st- and 2nd-contract exports into one app input file.

For every pair like DU1.csv + DU2.csv in the source folder, writes one file with:
    near = 1st contract PX_LAST (raw)
    far  = 2nd contract PX_LAST (raw)
    current_price = raw price of the contract held (far from the roll date until Bloomberg
                    switches, near otherwise), for % returns
    PX_CLOSE_1D = back-adjusted price (Panama / additive, as in Carver's
                  Advanced Futures Trading Strategies, Appendix B)
plus the 1st contract's other columns and metadata.

Roll dates follow Carver's calendar (pysystemtrade rollconfig.csv): roll on day ExpiryOffset of
the expiry month plus RollOffsetDays, e.g. Bund ~5 days before expiry, US Treasuries 25 days
before. Bloomberg's generic tickers only switch at expiry, so until then RX1 is still the old
contract and RX2 is the new one - both prices exist on Carver's roll date.

Back-adjustment: on the roll date the gap far - near is added to all prices up to and including
that day, so the adjusted price on the roll date equals the new contract's price. From then
until Bloomberg switches, the held contract is far (RX2); after the switch it's near (RX1).

Bloomberg's switch days are found from the jump in the 1st contract's open interest (the old
contract's OI has run down; the new one's is large). Volume is used as a check: a real switch
also shows a volume jump the same day. Rolls without one are marked in roll_dates/<name>_rolls.csv.
Volume is not used to find rolls on its own - it also spikes on news days and moves before the
switch. Instruments not in Carver's list roll on Bloomberg's switch.

Usage (from the project root):
    python tools/merge_near_far.py                              # DATA/Bonds_CSV -> DATA/Bonds_CSV/merged
    python tools/merge_near_far.py --src DATA/Bonds_CSV --out DATA/all_input_files
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

# 1st-contract metadata columns: Bloomberg puts them on the first row only
META_COLS = ['INSTRUMENT', 'SECTYPE', 'EXCHANGE', 'CRNCY', 'TICK_SIZE', 'TICK_VALUE', 'POINT_VALUE',
             'BID', 'ASK', 'EXECUTION', 'CLEARING', 'ASSET_CLASS', 'ROLL_PER_YEAR',
             'EXECUTION_COST', 'CLEARING_COST', 'ROLLS_PER_YEAR']

# Bloomberg names -> names the app reads
RENAME = {'EXECUTION': 'EXECUTION_COST', 'CLEARING': 'CLEARING_COST', 'ROLL_PER_YEAR': 'ROLLS_PER_YEAR'}

# Metadata the app needs (CONTRACT_VALUE and Exchange rate are not included here)
APP_META = ['INSTRUMENT', 'SECTYPE', 'EXCHANGE', 'CRNCY', 'TICK_SIZE', 'TICK_VALUE', 'POINT_VALUE',
            'EXECUTION_COST', 'CLEARING_COST', 'ROLLS_PER_YEAR']

# A switch to a new contract shows as open interest jumping by at least this factor
OI_JUMP = 1.5
# A switch confirmed by volume shows volume jumping by at least this factor the same day
VOLUME_JUMP = 3
# far must be the same instrument as near: if they differ by more than this (median), the 2nd
# file is probably the wrong ticker (e.g. 'IK2 Index' instead of 'IK2 Comdty')
MAX_MEDIAN_SPREAD = 0.05
# Rolls closer together than this (business days) are treated as one; keep the biggest jump
MIN_ROLL_SPACING = 40
# Default when the file has no ROLLS_PER_YEAR (the bond futures here are quarterly)
DEFAULT_ROLLS_PER_YEAR = 4

# Max business days a missing far price is carried forward
FAR_FILL_LIMIT = 3

# Carver's roll calendar (pysystemtrade data/futures/csvconfig/rollconfig.csv):
# Bloomberg root -> (Carver's name, ExpiryOffset day of the expiry month, RollOffsetDays)
CARVER_ROLLS = {
    'DU': ('SHATZ', 6, -5), 'OE': ('BOBL', 6, -5), 'RX': ('BUND', 6, -5), 'UB': ('BUXL', 6, -5),
    'OAT': ('OAT', 6, -5), 'IK': ('BTP', 6, -5),
    'TU': ('US2', 19, -25), 'FV': ('US5', 19, -25), 'TY': ('US10', 19, -25), 'US': ('US20', 19, -25),
    'JB': ('JGB', 15, -20), 'KAA': ('KR10', 18, -1), 'KE': ('KR3', 18, -1),
}


def read_bbg(path: str) -> pd.DataFrame:
    """Read a Bloomberg CSV, skipping any preamble rows above the 'Dates' header."""
    with open(path, encoding='utf-8-sig') as f:
        lines = f.readlines()
    header = next(i for i, line in enumerate(lines) if line.split(',')[0].strip() in ('Dates', 'Date'))
    df = pd.read_csv(path, skiprows=header, encoding='utf-8-sig', low_memory=False)
    if 'PX_LAST.1' in df.columns:
        raise ValueError(f'{os.path.basename(path)} holds several tickers in one sheet - re-export it alone')
    df = df.rename(columns={'Dates': 'Date'})
    # dd/mm/yyyy in some exports, yyyy-mm-dd in others
    iso = df['Date'].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}')
    dates = pd.Series(pd.NaT, index=df.index)
    dates[iso] = pd.to_datetime(df.loc[iso, 'Date'], format='mixed', errors='coerce')
    dates[~iso] = pd.to_datetime(df.loc[~iso, 'Date'], dayfirst=True, format='mixed', errors='coerce')
    df['Date'] = dates
    return df.dropna(subset=['Date'])


def detect_switches(oi: pd.Series, far: pd.Series) -> list:
    """Positions of the first day on each new contract, from jumps in open interest."""
    oi = pd.Series(oi.to_numpy(dtype=float))
    ratio = (oi / oi.shift()).where(pd.Series(far.to_numpy()).shift().notna())
    accepted = []
    for i in ratio[ratio >= OI_JUMP].sort_values(ascending=False).index:
        if all(abs(i - j) >= MIN_ROLL_SPACING for j in accepted):
            accepted.append(i)
    return sorted(accepted)


def back_adjust(near: pd.Series, far: pd.Series, rolls: list):
    """Panama back-adjustment.

    rolls: (roll_date, switch) positions per roll - roll_date = the day we roll (gap taken on its
    close), switch = Bloomberg's first day on the new contract. A plain int t means roll_date = t-1
    (roll on Bloomberg's own switch). Between the two the held contract is far.

    Returns (adjusted prices, gap on each roll date or 0, raw price of the held contract)."""
    near = pd.Series(near.to_numpy(dtype=float))
    far = pd.Series(far.to_numpy(dtype=float))
    gap = pd.Series(0.0, index=near.index)
    held = near.copy()
    for roll in rolls:
        roll_date, switch = (roll - 1, roll) if isinstance(roll, (int, np.integer)) else roll
        gap.iloc[roll_date] = far.iloc[roll_date] - near.iloc[roll_date]
        held.iloc[roll_date + 1:switch] = far.iloc[roll_date + 1:switch]
    # each price gets the sum of the gaps on its own roll date and every later one
    adjusted = held + gap[::-1].cumsum()[::-1]
    # from the roll date's close we hold the new contract
    current = held.copy()
    for roll in rolls:
        roll_date = roll - 1 if isinstance(roll, (int, np.integer)) else roll[0]
        current.iloc[roll_date] = far.iloc[roll_date]
    return adjusted.round(10).to_numpy(), gap.round(10).to_numpy(), current.to_numpy()


def carver_roll_dates(dates: pd.Series, far: pd.Series, switches: list, root: str):
    """Move each roll to Carver's calendar date. Returns [(roll_date, switch)] positions and the
    number of rolls left on Bloomberg's date (far price missing in the window).

    Carver's date = day ExpiryOffset of the expiry month + RollOffsetDays; we roll on the first
    trading day on or after it (closest match to his BUND roll calendar), but not before the
    previous contract switch and not after Bloomberg's own switch."""
    if root not in CARVER_ROLLS:
        return [(t - 1, t) for t in switches], 0
    _, expiry_day, offset = CARVER_ROLLS[root]
    rolls, fallback, prev = [], 0, 0
    for t in switches:
        # Old contract's expiry month: the quarter month (Mar/Jun/Sep/Dec) on or before its last day
        # in the data. Month-end expiries (US 2y/5y) can show up as a switch in the next month.
        last = dates.iloc[t - 1]
        month_index = last.year * 12 + last.month - 1
        while month_index % 12 + 1 not in (3, 6, 9, 12):
            month_index -= 1
        year, month = divmod(month_index, 12)
        target = pd.Timestamp(year, month + 1, expiry_day) + pd.Timedelta(days=offset)
        e = int(dates.searchsorted(target, side='left'))  # first trading day on or after target
        e = min(max(e, prev + 1), t - 1)
        if far.iloc[e:t].isna().any():  # need far from the roll date until the switch
            e, fallback = t - 1, fallback + 1
        rolls.append((e, t))
        prev = t
    return rolls, fallback


def merge_pair(first_path: str, second_path: str):
    first = read_bbg(first_path)
    second = read_bbg(second_path)

    meta = {c: first[c].dropna().iloc[0] for c in META_COLS if c in first.columns and first[c].notna().any()}

    first = first.dropna(subset=['PX_LAST']).set_index('Date').sort_index()
    # Only PX_LAST from the 2nd contract: its OHLC/volume/OI are often stale
    far = second.dropna(subset=['PX_LAST']).set_index('Date').sort_index()['PX_LAST']
    far = far[~far.index.duplicated(keep='last')]

    out = pd.DataFrame(index=first.index)
    out['PX_CLOSE_1D'] = first['PX_LAST']
    out['near'] = first['PX_LAST']
    out['far'] = far.reindex(out.index).ffill(limit=FAR_FILL_LIMIT)
    for c in ['PX_SETTLE', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'BLOOMBERG_CLOSE_PRICE', 'PX_VOLUME', 'OPEN_INT']:
        if c in first.columns:
            out[c] = first[c]
    for c, v in meta.items():
        out[RENAME.get(c, c)] = v

    if out.empty:
        raise ValueError('no PX_LAST prices in the 1st-contract file')
    spread = ((out['far'] - out['near']).abs() / out['near'].abs()).median()
    if spread > MAX_MEDIAN_SPREAD:
        raise ValueError(f'2nd-contract prices are {spread:.0%} away from the 1st contract - '
                         f'{os.path.basename(second_path)} looks like a different instrument (check the ticker)')
    out = out.reset_index()

    rolls = pd.DataFrame()
    fallback = 0
    if 'OPEN_INT' in out.columns and out['OPEN_INT'].notna().any():
        switches = [t for t in detect_switches(out['OPEN_INT'], out['far']) if pd.notna(out['far'].iloc[t - 1])]
        root = re.match(r'^(.*?)1', os.path.basename(first_path)).group(1).strip().upper()
        roll_pairs, fallback = carver_roll_dates(out['Date'], out['far'], switches, root)
        out['PX_CLOSE_1D'], out['roll_gap'], out['current_price'] = back_adjust(out['near'], out['far'], roll_pairs)
        roll_rows = [e for e, _ in roll_pairs]
        rolls = pd.DataFrame({
            'roll_date': out['Date'].iloc[roll_rows].dt.strftime('%d/%m/%Y').to_numpy(),
            'bloomberg_switch': out['Date'].iloc[switches].dt.strftime('%d/%m/%Y').to_numpy(),
            'trading_days_before_switch': [t - e for e, t in roll_pairs],
            'old_contract (near)': out['near'].iloc[roll_rows].to_numpy(),
            'new_contract (far)': out['far'].iloc[roll_rows].to_numpy(),
            'gap': out['roll_gap'].iloc[roll_rows].to_numpy(),
            'oi_jump': (out['OPEN_INT'] / out['OPEN_INT'].shift()).iloc[switches].round(2).to_numpy(),
        })
        if 'PX_VOLUME' in out.columns:
            vol_jump = (out['PX_VOLUME'] / out['PX_VOLUME'].shift()).iloc[switches].round(2).to_numpy()
            rolls['volume_jump'] = vol_jump
            rolls['confirmed_by_volume'] = vol_jump >= VOLUME_JUMP

    if 'current_price' in out.columns:
        cols = list(out.columns)
        cols.insert(cols.index('far') + 1, cols.pop(cols.index('current_price')))
        out = out[cols]
    out['Date'] = out['Date'].dt.strftime('%d/%m/%Y')
    rolls.attrs['fallback'] = fallback
    return out, rolls


def missing_rolls(df: pd.DataFrame, rolls: pd.DataFrame) -> list:
    """Stretches longer than 1.5x the normal roll spacing with no roll found."""
    per_year = pd.to_numeric(df.get('ROLLS_PER_YEAR', pd.Series([DEFAULT_ROLLS_PER_YEAR])), errors='coerce').iloc[0]
    limit = 1.5 * 252 / (per_year if per_year and per_year > 0 else DEFAULT_ROLLS_PER_YEAR)
    edges = [0] + df.index[df['Date'].isin(rolls['bloomberg_switch'])].tolist() + [len(df) - 1]
    return [f"{df['Date'].iloc[a]} -> {df['Date'].iloc[b]}" for a, b in zip(edges, edges[1:]) if b - a > limit]


def find_pairs(src: str):
    """Yield (name, first_path, second_path) for each X1*.csv that has a matching X2.csv."""
    files = {f.lower(): f for f in os.listdir(src) if f.lower().endswith('.csv')}
    for f in sorted(files.values()):
        m = re.match(r'^(.*?)1(_[^.]*)?\.csv$', f, re.IGNORECASE)
        if not m:
            continue
        second = files.get(f'{m.group(1)}2.csv'.lower())
        if second:
            yield f[:-4], os.path.join(src, f), os.path.join(src, second)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default=os.path.join('DATA', 'Bonds_CSV'))
    p.add_argument('--out', default=os.path.join('DATA', 'Bonds_CSV', 'merged'))
    p.add_argument('--overwrite', action='store_true', help='replace files that already exist in --out')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name, first_path, second_path in find_pairs(args.src):
        dest = os.path.join(args.out, f'{name}.csv')
        if os.path.exists(dest) and not args.overwrite:
            print(f'SKIP  {name}: {dest} exists (use --overwrite)')
            continue
        try:
            df, rolls = merge_pair(first_path, second_path)
        except Exception as e:
            print(f'FAIL  {name}: {e}')
            continue
        try:
            df.to_csv(dest, index=False)
        except PermissionError:
            print(f'FAIL  {name}: {dest} is open in another program (Excel?) - close it and rerun')
            continue
        if rolls.empty:
            note_rolls = '\n      no OPEN_INT: PX_CLOSE_1D is NOT back-adjusted (raw near)'
        else:
            roll_dir = os.path.join(args.out, 'roll_dates')
            os.makedirs(roll_dir, exist_ok=True)
            rolls.to_csv(os.path.join(roll_dir, f'{name}_rolls.csv'), index=False)
            root = re.match(r'^(.*?)1', os.path.basename(first_path)).group(1).strip().upper()
            schedule = (f"Carver's {CARVER_ROLLS[root][0]} calendar" if root in CARVER_ROLLS
                        else "Bloomberg's switch dates (not in Carver's list)")
            note_rolls = f'\n      {len(rolls)} rolls back-adjusted on {schedule}'
            if rolls.attrs.get('fallback'):
                note_rolls += f" ({rolls.attrs['fallback']} on Bloomberg's date: far price missing)"
            if 'confirmed_by_volume' in rolls.columns:
                unconfirmed = rolls.loc[~rolls['confirmed_by_volume'], 'bloomberg_switch'].tolist()
                note_rolls += f', {len(rolls) - len(unconfirmed)} confirmed by volume'
                if unconfirmed:
                    note_rolls += f' (check: {", ".join(unconfirmed[:5])}{" ..." if len(unconfirmed) > 5 else ""})'
            gaps = missing_rolls(df, rolls)
            if gaps:
                note_rolls += f'; no roll found in {len(gaps)} stretch(es): ' + '; '.join(gaps[:4])
                note_rolls += ' ...' if len(gaps) > 4 else ''
        missing_far = int(df['far'].isna().sum())
        note = f', {missing_far} rows without far' if missing_far else ''
        no_meta = [c for c in APP_META if c not in df.columns]
        if no_meta:
            note += '\n      missing metadata: ' + ', '.join(no_meta)
        print(f'OK    {name}: {len(df)} rows {df["Date"].iloc[0]} -> {df["Date"].iloc[-1]}{note}{note_rolls}')


if __name__ == '__main__':
    main()
