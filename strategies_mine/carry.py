# carry_run.py  (or use as the body of strategies_mine/carry.py if you prefer a script)
import os, numpy as np, pandas as pd

# ======== CONFIG (change per instrument/run) ========
IN_DIR     = r"C:\Users\loci_\Desktop\trading_webapp\DATA\all_input_files"
OUT_DIR    = r"C:\Users\loci_\Desktop\trading_webapp\DATA\all_output_files"
INSTRUMENT = "AD1_small.csv"   # e.g. "RX1_small.csv"
OUT_PREFIX = "AD1"             # e.g. "RX1"
DISTANCE   = 1/12              # AD1 monthly → 1/12 ; RX1 quarterly → 3/12

TRADING_DAYS     = 256
CAP              = 20.0
LOOKBACK_N       = 36                # std-dev decay lookback
ALPHA            = 2 / (LOOKBACK_N + 1)
FORECAST_SCALER  = 30.0              # carry scaler

os.makedirs(OUT_DIR, exist_ok=True)

def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns; tried {candidates} | found: {list(df.columns)}")

def _ewma_var_from_squared(sq: pd.Series, alpha: float) -> pd.Series:
    """EWMA on squared returns; first value = first square return (your spec)."""
    arr = sq.to_numpy()
    n   = len(arr)
    out = np.full(n, np.nan)
    # first non-nan
    mask = ~np.isnan(arr)
    if not mask.any():
        return pd.Series(out, index=sq.index)
    i0 = np.argmax(mask)
    out[i0] = arr[i0]
    prev = out[i0]
    for i in range(i0 + 1, n):
        x2 = 0.0 if np.isnan(arr[i]) else arr[i]
        prev = alpha * x2 + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=sq.index)

# ======== LOAD ========
path = os.path.join(IN_DIR, INSTRUMENT)
df   = pd.read_csv(path)
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

near_col = _pick_col(df, ["near","NEAR"])
far_col  = _pick_col(df, ["far","FAR"])
px_col   = _pick_col(df, ["PX_CLOSE_1D","px_close_1d","Close","close","PX_CLOSE","px_close"])

df["near"] = df[near_col].astype(float)
df["far"]  = df[far_col].astype(float)
px         = df[px_col].astype(float)

# ======== CARRY (your exact spec) ========
# return = Δ near ; square_returns = return^2
df["return"]          = df["near"].diff()
df["square_returns"]  = df["return"] ** 2

# variance via EWMA on squared returns (decay = 2/(N+1)), first = first square return
var    = _ewma_var_from_squared(df["square_returns"], ALPHA)
ann_sd = np.sqrt(var) * np.sqrt(TRADING_DAYS)

# price_difference = far - near ; net_expected_return = price_difference / DISTANCE
df["price_difference"]    = df["far"] - df["near"]
df["net_expected_return"] = df["price_difference"] / float(DISTANCE)

# raw_carry = net_expected_return / annual_standard_deviation
den = pd.Series(ann_sd, index=df.index).replace(0.0, np.nan)
raw_carry = df["net_expected_return"] / den

# forecast (scaled & capped)
df["forecast_carry"] = (raw_carry * FORECAST_SCALER).clip(-CAP, CAP)

# ======== RAW PnL proxy & Sharpe ========
# today’s forecast × tomorrow’s price_difference (as we agreed)
raw_pnl = df["forecast_carry"] * df["price_difference"].shift(-1)
mu, sd  = raw_pnl.mean(), raw_pnl.std()
sharpe  = np.nan if (sd is None or sd == 0 or np.isnan(sd)) else (mu / sd) * np.sqrt(TRADING_DAYS)

print(f"{OUT_PREFIX} Carry raw Sharpe: {sharpe:.3f}")

# ======== SAVE ========
out = df[["Date","near","far",px_col,"forecast_carry"]].copy()
out.rename(columns={px_col: "PX_CLOSE_1D"}, inplace=True)
out_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_CARRY.csv")
out.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
