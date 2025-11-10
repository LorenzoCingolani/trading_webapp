import os
import numpy as np
import pandas as pd

TRADING_DAYS = 256
CAP = 20.0
DECAY_N = 36
ALPHA = 2 / (DECAY_N + 1)

# fast -> scaler (slow = fast*4)
FORECAST_SCALERS = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}

IN_DIR    = r"C:\Users\loci_\Desktop\trading_webapp\DATA\all_input_files"
OUT_DIR   = r"C:\Users\loci_\Desktop\trading_webapp\DATA\all_output_files"
INSTRUMENT = "RX1_small.csv"     # change per run
OUT_PREFIX = "RX1"               # change per run

os.makedirs(OUT_DIR, exist_ok=True)

def _pick_price_col(df: pd.DataFrame) -> str:
    candidates = ["PX_CLOSE_1D", "px_close_1d", "PX_CLOSE", "px_close", "Close", "close", "price", "Price"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"No price column found. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )

def _ewma_vol_from_net_returns(px: pd.Series, alpha: float) -> pd.Series:
    """EWMA variance on *net* returns (price diffs), first = first squared diff."""
    diff = px.diff().to_numpy()
    n = len(diff)
    var = np.full(n, np.nan)
    # first non-nan
    mask = ~np.isnan(diff)
    if not mask.any():
        return pd.Series(var, index=px.index)
    i0 = np.argmax(mask)
    var[i0] = diff[i0] ** 2
    prev = var[i0]
    for i in range(i0 + 1, n):
        x = 0.0 if np.isnan(diff[i]) else diff[i]
        prev = alpha * (x * x) + (1 - alpha) * prev
        var[i] = prev
    return pd.Series(np.sqrt(var), index=px.index)  # return stdev

def compute_all_ewma(df_in: pd.DataFrame, crosses: dict, cap: float = 20.0) -> pd.DataFrame:
    df = df_in.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    px_col = _pick_price_col(df)
    px = df[px_col].astype(float)
    # returns for raw PnL proxy
    ret_pct = px.pct_change()

    # EWMA stdev from net returns (Carver)
    vol = _ewma_vol_from_net_returns(px, ALPHA).replace(0.0, np.nan)

    for (fast, slow), scaler in crosses.items():
        label = f"{fast}d_{slow}d"
        ewf = px.ewm(span=fast, adjust=False).mean()
        ews = px.ewm(span=slow, adjust=False).mean()
        raw_cross = ewf - ews
        vol_adj = raw_cross / vol
        fcast = (vol_adj * scaler).clip(-cap, cap)

        df[f"ewma_{label}_forecast"]   = fcast
        df[f"ewma_{label}_fcastxret"]  = fcast.shift(1) * ret_pct

    # keep a canonical price column in the output
    df.rename(columns={px_col: "PX_CLOSE_1D"}, inplace=True)
    return df

if __name__ == "__main__":
    path = os.path.join(IN_DIR, INSTRUMENT)
    df_raw = pd.read_csv(path)

    df_out = compute_all_ewma(df_raw, FORECAST_SCALERS, cap=CAP)
    out_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_EWMA_ALL.csv")
    df_out.to_csv(out_path, index=False)

    print(f"Picked price column: 'PX_CLOSE_1D'")
    print("\n=== EWMA Sharpe (raw: forecast[t-1] × ret_pct[t]) ===")
    for (fast, slow), _ in FORECAST_SCALERS.items():
        col = f"ewma_{fast}d_{slow}d_fcastxret"
        s = df_out[col].dropna()
        sh = np.nan
        if len(s) > 2 and s.std() not in (0, np.nan):
            sh = (s.mean() / s.std()) * np.sqrt(TRADING_DAYS)
        print(f"{fast:>2d}/{slow:<3d}  Sharpe = {sh:6.3f}")

    print("\n✅ Saved:", out_path)
