# compute_pdm_and_portfolio.py
import os, numpy as np, pandas as pd

BASE   = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
OUTDIR = os.path.join(BASE, "all_output_files")

# instrument -> weight
INSTR_WEIGHTS = {"RX1": 0.5, "AD1": 0.5}

TRADING_DAYS = 256
CAP_PDM = 2.5

def _ann_sharpe(x, td=TRADING_DAYS):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd==0 or np.isnan(sd)) else (mu/sd)*np.sqrt(td)

def _price_col(df):
    for c in ["PX_CLOSE_1D","px_close_1d","Close","close"]:
        if c in df.columns: return c
    raise KeyError("Price column not found.")

comb_paths = {inst: os.path.join(OUTDIR, f"{inst}_COMBINED.csv") for inst in INSTR_WEIGHTS}
raw_series = {}
for inst, p in comb_paths.items():
    if not os.path.exists(p):
        print(f"[{inst}] Missing {p}; skipping.")
        continue
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    px = df[_price_col(df)].astype(float)
    ret_pct = px.pct_change()
    raw = df["forecast_combined"].shift(1) * ret_pct
    raw_series[inst] = raw.rename(inst)

# Align & compute PDM
panel = pd.concat(raw_series.values(), axis=1).dropna()
if panel.empty:
    print("No overlap across instruments.")
    raise SystemExit

C = panel.corr().values
w = np.array([INSTR_WEIGHTS[c] for c in panel.columns]).reshape(-1,1)
denom = float(w.T @ C @ w)
pdm = 1.0 / np.sqrt(denom) if denom>0 and not np.isnan(denom) else 1.0
pdm = float(min(pdm, CAP_PDM))

# Portfolio raw series (apply PDM)
port = sum(INSTR_WEIGHTS[c]*panel[c] for c in panel.columns) * pdm
sh = _ann_sharpe(port)

print(f"PDM={pdm:.3f} | Portfolio raw Sharpe={sh:.3f} | Instruments={list(panel.columns)}")

# Save summary
summary = pd.DataFrame({
    "instrument": list(panel.columns)+["PORTFOLIO"],
    "weight": [INSTR_WEIGHTS[c] for c in panel.columns]+[sum(INSTR_WEIGHTS.values())],
    "sharpe": [ _ann_sharpe(panel[c]) for c in panel.columns ] + [sh],
    "pdm":    [np.nan]*len(panel.columns) + [pdm],
})
summary.to_csv(os.path.join(OUTDIR,"PORTFOLIO_pdm_summary.csv"), index=False)
