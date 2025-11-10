# combine_forecast.py
# Dynamically computes FDM (Carver) across forecasts for each instrument,
# with 60% weight to EWMA (split equally across Good_trade speeds) and 40% to Carry.
# Combined forecast = (weights · forecasts) * FDM, capped ±20.

import os
import numpy as np
import pandas as pd

# ====== CONFIG ======
BASE   = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
OUTDIR = os.path.join(BASE, "all_output_files")
INSTRUMENTS = ["RX1", "AD1"]      # extend as needed

CAP_FORECAST = 20.0
TRADING_DAYS = 256
FDM_CAP      = 2.5

# ====== HELPERS ======
def _pick_price_col(df):
    for c in ["PX_CLOSE_1D", "px_close_1d", "Close", "close"]:
        if c in df.columns: return c
    raise KeyError("Price column not found in dataframe.")

def _ann_sharpe(x, td=TRADING_DAYS):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu / sd) * np.sqrt(td)

def _fdm_from_rawpnl(rawp_df: pd.DataFrame, weights: np.ndarray, cap=FDM_CAP):
    """
    rawp_df: columns = strategies, values = forecast.shift(1) * ret_pct
    weights: 1D np array aligned to columns(rawp_df)
    """
    mat = rawp_df.dropna()
    if mat.shape[0] < 30 or mat.shape[1] == 0:
        return 1.0, None, np.nan
    C = mat.corr()
    w = weights.reshape(-1, 1)
    denom = float((w.T @ C.values @ w)[0, 0])
    if denom <= 0 or np.isnan(denom):
        return 1.0, C, np.nan
    fdm = 1.0 / np.sqrt(denom)
    fdm = float(min(fdm, cap))
    # average off-diagonal correlation (diagnostic)
    if C.shape[0] > 1:
        offdiag = C.values.copy()
        np.fill_diagonal(offdiag, np.nan)
        avg_corr = np.nanmean(offdiag)
    else:
        avg_corr = np.nan
    return fdm, C, avg_corr

def _compute_strategy_weights(strat_cols):
    """
    60% total to EWMA_* (equal split), 40% to 'carry' if present.
    Edge cases handled (only EWMA or only carry).
    Returns np.array aligned with strat_cols.
    """
    ewma_idx = [i for i, c in enumerate(strat_cols) if str(c).startswith("EWMA_")]
    has_carry = any(str(c).lower() == "carry" for c in strat_cols)
    n_ewma = len(ewma_idx)

    if has_carry and n_ewma > 0:
        w = np.zeros(len(strat_cols), dtype=float)
        for i in ewma_idx:
            w[i] = 0.60 / n_ewma
        carry_pos = [i for i, c in enumerate(strat_cols) if str(c).lower() == "carry"][0]
        w[carry_pos] = 0.40
        return w

    if n_ewma > 0 and not has_carry:
        w = np.zeros(len(strat_cols), dtype=float)
        for i in ewma_idx:
            w[i] = 1.0 / n_ewma
        return w

    if has_carry and n_ewma == 0:
        w = np.zeros(len(strat_cols), dtype=float)
        carry_pos = [i for i, c in enumerate(strat_cols) if str(c).lower() == "carry"][0]
        w[carry_pos] = 1.0
        return w

    return np.ones(len(strat_cols)) / max(1, len(strat_cols))

# ====== MAIN LOOP ======
os.makedirs(OUTDIR, exist_ok=True)

for inst in INSTRUMENTS:
    # 1) Load Carry (optional)
    carry_path = os.path.join(OUTDIR, f"{inst}_CARRY_timeseries.csv")
    carry_df = None
    if os.path.exists(carry_path):
        carry_df = pd.read_csv(carry_path)
        carry_df["Date"] = pd.to_datetime(carry_df["Date"], dayfirst=True, errors="coerce")
        carry_df = carry_df.dropna(subset=["Date"]).sort_values("Date")

    # 2) Which EWMA speeds passed cost filter
    ewma_good = []
    cost_sum = os.path.join(OUTDIR, f"{inst}_EWMA_cost_filter_summary.csv")
    if os.path.exists(cost_sum):
        cs = pd.read_csv(cost_sum)
        ewma_good = cs.loc[cs["decision"] == "Good_trade", "strategy"].tolist()  # e.g. "EWMA_4/16"

    # 3) Load only the chosen EWMA forecasts (Date + forecast col only)
    ewma_list = []  # list of (label, df[["Date", label]])
    for s in ewma_good:
        try:
            fs = s.split("_")[1]  # "4/16"
            fast, slow = fs.split("/")
            p = os.path.join(OUTDIR, f"{inst}_EWMA_{fast}d_{slow}d_timeseries.csv")
            if not os.path.exists(p):
                continue
            edf = pd.read_csv(p)
            edf["Date"] = pd.to_datetime(edf["Date"], dayfirst=True, errors="coerce")
            edf = edf.dropna(subset=["Date"]).sort_values("Date")
            fcol = "forecast" if "forecast" in edf.columns else next(
                (c for c in edf.columns if c.startswith("forecast")), None
            )
            if fcol is None:
                continue
            label = f"EWMA_{fast}/{slow}"
            ewma_list.append((label, edf[["Date", fcol]].rename(columns={fcol: label})))
        except Exception:
            continue

    if (carry_df is None) and (len(ewma_list) == 0):
        print(f"[{inst}] Nothing to combine (no carry and no Good_trade EWMA).")
        continue

    # 4) Build base: Date + PX + ret_pct from whichever has price
    # Prefer carry; else, grab price from one EWMA timeseries file
    if carry_df is not None:
        px_col = _pick_price_col(carry_df)
        base = carry_df[["Date", px_col]].rename(columns={px_col: "PX_CLOSE_1D"}).copy()
    else:
        # load one EWMA timeseries again to get PX column
        lbl, _ = ewma_list[0]
        fast, slow = lbl.split("_")[1].split("/")
        p0 = os.path.join(OUTDIR, f"{inst}_EWMA_{fast}d_{slow}d_timeseries.csv")
        edf0 = pd.read_csv(p0)
        edf0["Date"] = pd.to_datetime(edf0["Date"], dayfirst=True, errors="coerce")
        edf0 = edf0.dropna(subset=["Date"]).sort_values("Date")
        px_col0 = _pick_price_col(edf0)
        base = edf0[["Date", px_col0]].rename(columns={px_col0: "PX_CLOSE_1D"}).copy()

    base["ret_pct"] = pd.to_numeric(base["PX_CLOSE_1D"], errors="coerce").pct_change()

    # 5) Merge forecasts (ONLY forecast columns)
    strat_cols = []
    if carry_df is not None:
        fcol = "forecast" if "forecast" in carry_df.columns else next(
            (c for c in carry_df.columns if c.startswith("forecast")), None
        )
        tmp = carry_df[["Date", fcol]].rename(columns={fcol: "carry"})
        base = pd.merge(base, tmp, on="Date", how="inner")
        strat_cols.append("carry")

    for lbl, edf_fc in ewma_list:
        base = pd.merge(base, edf_fc, on="Date", how="inner")  # adds a column named lbl
        strat_cols.append(lbl)

    if len(strat_cols) == 0:
        print(f"[{inst}] No strategy columns after merge.")
        continue

    # 6) Weights (60% EWMA / 40% Carry, equal within EWMA)
    weights = _compute_strategy_weights(strat_cols)

    # 7) Build raw PnL proxy per strategy: forecast.shift(1) * ret_pct
    rawp = pd.DataFrame(index=base.index)
    for col in strat_cols:
        rawp[col] = pd.to_numeric(base[col], errors="coerce").shift(1) * base["ret_pct"]

    # 8) FDM from correlations of rawP&L
    fdm, corr_mat, avg_corr = _fdm_from_rawpnl(rawp[strat_cols], weights, cap=FDM_CAP)

    # 9) Combined forecast (apply weights then FDM, finally cap ±20)
    fc_mat = base[strat_cols].to_numpy()
    f_comb = (fc_mat @ weights).flatten() * fdm
    f_comb = np.clip(f_comb, -CAP_FORECAST, CAP_FORECAST)
    base["forecast_combined"] = f_comb

    # 10) Diagnostics and save
    comp_raw = (pd.to_numeric(base["forecast_combined"], errors="coerce").shift(1) * base["ret_pct"])
    sh = _ann_sharpe(comp_raw)

    # Pretty weights print
    w_map = {name: float(weights[i]) for i, name in enumerate(strat_cols)}
    print(f"[{inst}] FDM={fdm:.3f} (cap {FDM_CAP}) | avg off-diag corr={avg_corr:.3f} | "
          f"Combined raw Sharpe={sh:.3f} | strategies={strat_cols} | weights={w_map}")

    # Save combined and components
    out_cols = ["Date", "PX_CLOSE_1D", "forecast_combined"] + strat_cols
    out_path = os.path.join(OUTDIR, f"{inst}_COMBINED.csv")
    base[out_cols].to_csv(out_path, index=False)

    # Save a small summary with FDM and weights
    summ = pd.DataFrame([{
        "instrument": inst,
        "fdm": fdm,
        "avg_offdiag_corr": avg_corr,
        "strategies": ";".join(strat_cols),
        "weights": ";".join(f"{k}:{w_map[k]:.4f}" for k in strat_cols),
        "combined_raw_sharpe": sh,
        "obs_used": int(rawp[strat_cols].dropna().shape[0])
    }])
    summ_path = os.path.join(OUTDIR, f"{inst}_COMBINED_FDM_summary.csv")
    summ.to_csv(summ_path, index=False)

    # Optionally save correlation matrix (if computed)
    if corr_mat is not None:
        corr_path = os.path.join(OUTDIR, f"{inst}_COMBINED_FDM_corr.csv")
        corr_mat.round(4).to_csv(corr_path)
