"""Weekly volatility-normalised subsystem returns, one column per instrument, for the Carver
handcrafting step.

Reads each instrument's combined forecast from DATA/combinedForecast/{instrument}.csv (already
produced by Validation: FDM-blended EWMA + Carry, capped at +/-20) and computes, per day t:

    r_t = (F_{t-1} / 10) * (dP_t / sigma_{t-1})

where sigma is the 20-day simple price volatility from steps/volatility.py. The sum of daily
returns over each week (ending Friday) is written to DATA/portfolio_weights/weekly_subsystem_returns.csv.

Weeks with no data for an instrument stay NaN rather than 0, so an instrument's history starts at
its own inception and doesn't drag its weight through the pre-inception weeks.

Each weekly column is then divided by its own full-sample standard deviation, so every instrument
enters the handcrafting step at unit volatility. This scaling is IN-SAMPLE (it uses the whole
history), and the weights built from it are test/diagnostic only until subsystem returns come from
back-adjusted prices. The per-column scale factors are written alongside the output.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from steps.volatility import simple_price_volatility  # noqa: E402

FORECAST_DIVISOR = 10.0
INPUT_DIR = os.path.join(ROOT, 'DATA', 'input_instruments')
COMBINED_DIR = os.path.join(ROOT, 'DATA', 'combinedForecast')
OUTPUT_DIR = os.path.join(ROOT, 'DATA', 'portfolio_weights')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'weekly_subsystem_returns.csv')
SCALE_PATH = os.path.join(OUTPUT_DIR, 'weekly_subsystem_vol_scale.csv')
RAW_PATH = os.path.join(OUTPUT_DIR, 'weekly_subsystem_returns_raw.csv')


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors='coerce')


def daily_subsystem_returns(instrument: str) -> pd.Series:
    inp = pd.read_csv(os.path.join(INPUT_DIR, f'{instrument}.csv'))
    inp['Date'] = _parse_dates(inp['Date'])
    inp = inp.dropna(subset=['Date']).set_index('Date').sort_index()

    comb = pd.read_csv(os.path.join(COMBINED_DIR, f'{instrument}.csv'))
    comb['Date'] = _parse_dates(comb['Date'])
    comb = comb.dropna(subset=['Date']).set_index('Date').sort_index()

    px = pd.to_numeric(inp['PX_CLOSE_1D'], errors='coerce')
    df = pd.DataFrame({'px': px, 'sigma': simple_price_volatility(px)})
    df = df.join(comb['FinalForecast'].rename('forecast'), how='left')

    d_price = df['px'].diff()
    forecast_prev = df['forecast'].shift(1)
    sigma_prev = df['sigma'].shift(1)
    r = (forecast_prev / FORECAST_DIVISOR) * (d_price / sigma_prev)
    return r.rename(instrument)


def weekly_subsystem_returns(instruments: list[str]) -> pd.DataFrame:
    daily = pd.concat([daily_subsystem_returns(i) for i in instruments], axis=1)
    return daily.resample('W-FRI').sum(min_count=1)


def main() -> None:
    instruments = sorted(
        f[:-4] for f in os.listdir(INPUT_DIR)
        if f.endswith('.csv') and os.path.exists(os.path.join(COMBINED_DIR, f))
    )
    if not instruments:
        raise SystemExit('No instruments have both an input file and a combined forecast.')

    weekly = weekly_subsystem_returns(instruments)
    raw = weekly.copy()
    raw.index.name = 'Date'
    scale = weekly.std()
    standardised = weekly.div(scale, axis=1)
    standardised.index.name = 'Date'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw.to_csv(RAW_PATH, date_format='%Y-%m-%d')
    standardised.to_csv(OUTPUT_PATH, date_format='%Y-%m-%d')
    scale.rename('full_sample_std').to_csv(SCALE_PATH)
    print(f'Wrote {standardised.shape[0]} weeks x {standardised.shape[1]} instruments to {OUTPUT_PATH}')
    print('In-sample unit-vol scaling (test/diagnostic only until back-adjusted prices):')
    print(scale.to_string())


if __name__ == '__main__':
    main()
