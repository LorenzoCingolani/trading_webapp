"""Build app input files from Norgate Data (one file per instrument).

Reads DATA/instrument_settings.csv (one row per instrument):
    norgate_symbol  Norgate market symbol, e.g. FGBL, ZN, ES, CL, 6E
    file_name       name of the file to write (= instrument name in the app)
    spread_ticks    typical bid-ask spread in ticks (used for the cost calculation)
    EXECUTION_COST, CLEARING_COST   your broker's costs per contract, in the instrument's currency

and writes one CSV per instrument with the columns the app reads:
    PX_CLOSE_1D    Norgate's back-adjusted continuous price (&SYM_CCB)
    near           Norgate's raw continuous price (&SYM): the front contract
    far            the next contract's raw price, for carry (carry = far - near, unchanged)
    current_price  raw price of the contract held at each close, for % returns. Norgate switches
                   to the new contract at the close of the day before its delivery month changes,
                   so on that day it's the new contract's price.
    near_contract, far_contract   which contracts those are (e.g. FGBL-2026Z)
    metadata       CRNCY, EXCHANGE, SECTYPE, TICK_SIZE, TICK_VALUE, POINT_VALUE, ASSET_CLASS,
                   ROLLS_PER_YEAR, Exchange rate, QUOTE_CONVENTION, BID, ASK, costs, Standard Cost

Needs the Norgate Data Updater running and the norgatedata package (in the trading_env env).

Usage (from the project root, with trading_env's python):
    python tools/build_from_norgate.py                      # writes to DATA/all_input_files
    python tools/build_from_norgate.py --overwrite          # replace existing files
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.getLogger('norgatedata').setLevel(logging.ERROR)
import norgatedata as nd  # noqa: E402

SETTINGS = os.path.join('DATA', 'instrument_settings.csv')
OUT_DIR = os.path.join('DATA', 'all_input_files')
MONTH_CODES = 'FGHJKMNQUVXZ'
TRADING_DAYS = 256

ASSET_CLASSES = {'Interest Rate': 'BOND', 'Stock Index': 'EQUITY', 'Currency': 'FX', 'Energy': 'ENERGY',
                 'Metal': 'METAL', 'Grain': 'AGRICULTURE', 'Soft': 'AGRICULTURE', 'Meat': 'AGRICULTURE',
                 'Volatility': 'VOLATILITY'}


def contract_symbol(market: str, delivery_month: int) -> str:
    """202612 -> FGBL-2026Z"""
    year, month = divmod(int(delivery_month), 100)
    return f'{market}-{year}{MONTH_CODES[month - 1]}'


def contract_key(symbol: str) -> int:
    """FGBL-2026Z -> 202612, for sorting contracts by delivery."""
    year, code = symbol.rsplit('-', 1)[1][:4], symbol.rsplit('-', 1)[1][4]
    return int(year) * 100 + MONTH_CODES.index(code) + 1


def to_tick(x, tick):
    """Norgate stores prices as float32 (126.120003): snap to the tick grid."""
    return (np.round(np.asarray(x, dtype=float) / tick) * tick).round(10)


def exchange_rate(currency: str):
    """Latest rate to USD and the app's quote convention for it."""
    if currency == 'USD':
        return 1.0, 'XXXUSD'
    fx = set(nd.database_symbols('Forex Spot'))
    for pair, convention in ((f'{currency}USD', 'XXXUSD'), (f'USD{currency}', 'USDXXX')):
        if pair in fx:
            rate = nd.price_timeseries(pair, timeseriesformat='pandas-dataframe')['Close'].dropna().iloc[-1]
            return round(float(rate), 6), convention
    raise ValueError(f'no {currency}/USD rate in Norgate Forex Spot')


def build(market: str, spread_ticks: float, execution_cost: float, clearing_cost: float) -> pd.DataFrame:
    raw = nd.price_timeseries('&' + market, timeseriesformat='pandas-dataframe')
    ccb = nd.price_timeseries('&' + market + '_CCB', timeseriesformat='pandas-dataframe')['Close']
    tick, point_value = nd.tick_size('&' + market), nd.point_value('&' + market)

    df = pd.DataFrame(index=raw.index)
    df['PX_CLOSE_1D'] = to_tick(ccb.reindex(raw.index), tick)
    df['near'] = to_tick(raw['Close'], tick)
    df['near_contract'] = raw['Delivery Month'].astype(int).map(lambda m: contract_symbol(market, m))

    # contract prices, and the next contract after each one
    contracts = sorted(nd.futures_market_session_contracts(market), key=contract_key)
    closes = {c: pd.Series(to_tick(s, tick), index=s.index) for c in contracts
              for s in [nd.price_timeseries(c, timeseriesformat='pandas-dataframe')['Close']]}
    next_of = dict(zip(contracts, contracts[1:]))
    df['far_contract'] = df['near_contract'].map(next_of)
    df['far'] = [closes.get(c, pd.Series(dtype=float)).get(d, np.nan) if isinstance(c, str) else np.nan
                 for d, c in zip(df.index, df['far_contract'])]

    # held contract at each close: the new one from the close of the day before a switch
    df['current_price'] = df['near']
    switch = df['near_contract'] != df['near_contract'].shift()
    switch.iloc[0] = False
    for pos in np.flatnonzero(switch.to_numpy()):
        new, day_before = df['near_contract'].iloc[pos], df.index[pos - 1]
        df.loc[day_before, 'current_price'] = closes[new].get(day_before, np.nan)

    df['PX_OPEN'], df['PX_HIGH'], df['PX_LOW'] = (to_tick(raw[c], tick) for c in ('Open', 'High', 'Low'))
    df['PX_VOLUME'], df['OPEN_INT'] = raw['Volume'], raw['Open Interest'].replace(0, np.nan)

    currency = nd.currency('&' + market)
    rate, convention = exchange_rate(currency)
    held_months = {int(m) % 100 for m in raw['Delivery Month'].iloc[-504:]}  # last ~2 years
    last_price = df['current_price'].dropna().iloc[-1]

    meta = {
        'INSTRUMENT': nd.futures_market_name('&' + market), 'SECTYPE': 'Future',
        'EXCHANGE': nd.exchange_name('&' + market), 'CRNCY': currency,
        'TICK_SIZE': tick, 'TICK_VALUE': tick * point_value, 'POINT_VALUE': point_value,
        'BID': last_price, 'ASK': float(to_tick(last_price + spread_ticks * tick, tick)),
        'EXECUTION_COST': execution_cost, 'CLEARING_COST': clearing_cost,
        'ASSET_CLASS': ASSET_CLASSES.get(nd.subtype1('&' + market), nd.subtype1('&' + market)),
        'ROLLS_PER_YEAR': len(held_months),
        'Exchange rate': rate, 'QUOTE_CONVENTION': convention,
    }
    for k, v in meta.items():
        df[k] = v

    # Fallback only - the app recalculates this from BID/ASK/costs (p_pages/main_analysis_page.py
    # _calc_standard_cost); same formula, needed because the Settings check requires a number.
    st_dev = df['PX_CLOSE_1D'].diff().rolling(20).std().dropna().iloc[-1]
    one_way = spread_ticks * tick / 2 * point_value + execution_cost + clearing_cost
    df['Standard Cost'] = 2 * one_way / (st_dev * point_value * np.sqrt(TRADING_DAYS))

    df.index.name = 'Date'
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--settings', default=SETTINGS)
    p.add_argument('--out', default=OUT_DIR)
    p.add_argument('--overwrite', action='store_true', help='replace files that already exist in --out')
    args = p.parse_args()

    if not nd.status():
        raise SystemExit('Norgate Data Updater is not running - start it and try again')
    settings = pd.read_csv(args.settings)
    os.makedirs(args.out, exist_ok=True)

    for row in settings.itertuples(index=False):
        dest = os.path.join(args.out, f'{row.file_name}.csv')
        if os.path.exists(dest) and not args.overwrite:
            print(f'SKIP  {row.file_name}: {dest} exists (use --overwrite)')
            continue
        try:
            df = build(row.norgate_symbol, row.spread_ticks, row.EXECUTION_COST, row.CLEARING_COST)
            df.to_csv(dest, index=False)
        except PermissionError:
            print(f'FAIL  {row.file_name}: {dest} is open in another program (Excel?) - close it and rerun')
            continue
        except Exception as e:
            print(f'FAIL  {row.file_name}: {e}')
            continue
        missing_far = int(df['far'].isna().sum())
        print(f"OK    {row.file_name}: {df['INSTRUMENT'].iloc[0]}, {len(df)} rows {df['Date'].iloc[0]} -> "
              f"{df['Date'].iloc[-1]}, {df['CRNCY'].iloc[0]} (rate {df['Exchange rate'].iloc[0]:.4f}), "
              f"{int(df['ROLLS_PER_YEAR'].iloc[0])} rolls/yr" + (f', {missing_far} days without far' if missing_far else ''))


if __name__ == '__main__':
    main()
