# portfolio_final_framework.py
import os
import numpy as np
import pandas as pd

# ================== CONFIG ==================
BASE      = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
IN_DIR    = os.path.join(BASE, "all_input_files")
OUT_DIR   = os.path.join(BASE, "all_output_files")

# Instruments and portfolio weights
INSTRUMENTS = {
    "RX1": {"weight": 0.5, "input_file": "RX1_small.csv"},
    "AD1": {"weight": 0.5, "input_file": "AD1_small.csv"},
}

TRADING_DAYS       = 256
ANNUAL_TARGET_VOL  = 0.20
START_NAV          = 10_000_000.0
PDM_CAP            = 2.5
VAR_LOOKBACK       = 36
ALPHA              = 2.0 / (VAR_LOOKBACK + 1.0)  # EWMA variance decay

os.makedirs(OUT_DIR, exist_ok=True)

# ================== HELPERS ==================
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_dates(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

def ewma_stdev_net(px: pd.Series, alpha: float) -> pd.Series:
    """EWMA stdev on net returns (price diffs). First var = first diff^2."""
    diff = px.diff().to_numpy()
    n = len(diff)
    var = np.full(n, np.nan)
    mask = ~np.isnan(diff)
    if not mask.any():
        return pd.Series(np.sqrt(var), index=px.index)
    i0 = np.argmax(mask)
    var[i0] = diff[i0] ** 2
    prev = var[i0]
    for i in range(i0 + 1, n):
        x = 0.0 if np.isnan(diff[i]) else diff[i]
        prev = ALPHA * (x * x) + (1 - ALPHA) * prev
        var[i] = prev
    return pd.Series(np.sqrt(var), index=px.index)

def ann_sharpe(x):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu/sd) * np.sqrt(TRADING_DAYS)

def sortino(x):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    dd = s[s < 0].std()
    mu = s.mean()
    return np.nan if (dd == 0 or np.isnan(dd)) else (mu/dd) * np.sqrt(TRADING_DAYS)

def max_drawdown_from_cum(cum):
    s = pd.Series(cum).fillna(0.0)
    peak = s.cummax()
    dd = s - peak
    return dd.min()  # negative

def calmar_from_pnl(p):
    cum = p.cumsum()
    mdd = max_drawdown_from_cum(cum)
    ann_ret = p.mean() * TRADING_DAYS
    if (mdd is None) or np.isnan(mdd) or mdd == 0:
        return np.nan, mdd, ann_ret
    return ann_ret / abs(mdd), mdd, ann_ret

def compute_pdm_from_prices(aligned_price_series: dict, weights_dict: dict, cap=2.5):
    """
    PDM from price pct-change correlations: 1/sqrt(w^T C w), capped.
    aligned_price_series: {inst -> Series of PX_CLOSE_1D on common dates}
    """
    px_panel = pd.concat(aligned_price_series, axis=1).dropna()
    if px_panel.empty or px_panel.shape[1] == 0:
        return 1.0, list(aligned_price_series.keys()), None
    ret_panel = px_panel.pct_change().dropna()
    if ret_panel.empty:
        return 1.0, list(px_panel.columns), None
    C = ret_panel.corr()
    cols = list(C.columns)
    w = np.array([weights_dict[c] for c in cols], dtype=float).reshape(-1, 1)
    denom = float((w.T @ C.values @ w)[0, 0])
    if denom <= 0 or np.isnan(denom):
        return 1.0, cols, C
    pdm = 1.0 / np.sqrt(denom)
    return float(min(pdm, cap)), cols, C

# ================== LOAD COMBINED & INPUTS ==================
combo = {}
inputs = {}
for inst, cfg in INSTRUMENTS.items():
    comb_path = os.path.join(OUT_DIR, f"{inst}_COMBINED.csv")
    if not os.path.exists(comb_path):
        print(f"❌ Missing combined file for {inst}: {comb_path}")
        continue
    dfc = ensure_dates(pd.read_csv(comb_path))
    px_col = pick_col(dfc, ["PX_CLOSE_1D","px_close_1d","Close","close"])
    if px_col != "PX_CLOSE_1D":
        dfc = dfc.rename(columns={px_col: "PX_CLOSE_1D"})
    combo[inst] = dfc[["Date", "PX_CLOSE_1D", "forecast_combined"]].copy()

    # input file for POINT_VALUE/FX
    in_path = os.path.join(IN_DIR, cfg["input_file"])
    dfi = ensure_dates(pd.read_csv(in_path))

    fx_col  = pick_col(dfi, ["FX_TO_USD","fx_to_usd","FX","fx"])
    pt_col  = pick_col(dfi, ["POINT_VALUE","point_value","PointValue"])
    tv_col  = pick_col(dfi, ["TICK_VALUE","tickValue","tick_value"])
    ts_col  = pick_col(dfi, ["TICK_SIZE","tickSize","tick_size"])

    if pt_col:
        dfi["POINT_VALUE"] = pd.to_numeric(dfi[pt_col], errors="coerce")
    elif tv_col and ts_col:
        dfi["POINT_VALUE"] = pd.to_numeric(dfi[tv_col], errors="coerce") / pd.to_numeric(dfi[ts_col], errors="coerce")
    else:
        raise KeyError(f"{inst}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE) in {cfg['input_file']}")

    dfi["FX_TO_USD"] = pd.to_numeric(dfi[fx_col], errors="coerce") if fx_col else 1.0
    inputs[inst] = dfi[["Date","POINT_VALUE","FX_TO_USD"]].copy()

if not combo:
    raise SystemExit("No instruments loaded. Ensure *_COMBINED.csv exist.")

# ================== ALIGN DATES ACROSS INSTRUMENTS ==================
dates = None
for inst, df in combo.items():
    dates = df["Date"] if dates is None else pd.merge(pd.DataFrame({"Date": dates}), df[["Date"]], on="Date", how="inner")["Date"]
dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)

aligned = {}
for inst in combo:
    dfc = combo[inst]
    dfi = inputs[inst]
    m = pd.merge(pd.merge(pd.DataFrame({"Date": dates}), dfc, on="Date", how="left"),
                 dfi, on="Date", how="left").dropna(subset=["PX_CLOSE_1D"])
    m = m.sort_values("Date").reset_index(drop=True)
    aligned[inst] = m

# ================== PDM FROM PRICE CORRELATIONS ==================
aligned_px = {inst: pd.to_numeric(df["PX_CLOSE_1D"], errors="coerce") for inst, df in aligned.items()}
weights_map = {inst: INSTRUMENTS[inst]["weight"] for inst in aligned}
PDM, pdm_cols, corr_mat = compute_pdm_from_prices(aligned_px, weights_map, cap=PDM_CAP)

print("\n=== PDM (from price % changes) ===")
print(f"weights: {weights_map}")
print(f"Instruments used: {pdm_cols}")
print(f"PDM (cap {PDM_CAP}): {PDM:.3f}")
if corr_mat is not None:
    print("\nCorrelation matrix of % returns:")
    print(corr_mat.round(3).to_string())

# ================== PRECOMPUTE SIZING SCAFFOLD PER INSTRUMENT ==================
scaffold = {}
for inst, df in aligned.items():
    px = pd.to_numeric(df["PX_CLOSE_1D"], errors="coerce")
    fx = pd.to_numeric(df["FX_TO_USD"], errors="coerce").fillna(1.0)
    point_value = float(pd.to_numeric(df["POINT_VALUE"], errors="coerce").dropna().iloc[0])

    stdev = ewma_stdev_net(px, ALPHA)                 # EWMA stdev on net returns
    price_vol = (stdev / px) * 100.0
    price_vol = price_vol.round(2)                    # your convention
    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    icv_local = block_value * price_vol               # no /100 here
    ivv_usd = icv_local * fx                          # FX>1 → USD IVV higher

    scaffold[inst] = {
        "px": px, "fx": fx, "point_value": point_value,
        "stdev": stdev, "price_vol": price_vol, "one_pct_move": one_pct_move,
        "block_value": block_value, "icv_local": icv_local, "ivv_usd": ivv_usd
    }

# ================== PORTFOLIO SIMULATION (NAV-lag, weights, PDM) ==================
n = len(dates)
nav = np.zeros(n, dtype=float)
nav[0] = START_NAV

state = {}
for inst in aligned:
    state[inst] = {
        "dcvt": np.zeros(n, dtype=float),
        "vol_scaler": np.zeros(n, dtype=float),
        "subsys_pos": np.zeros(n, dtype=float),
        "target": np.zeros(n, dtype=int),
        "pos": np.zeros(n, dtype=float),
        "trades": np.zeros(n, dtype=float),
        "trade_pnl": np.zeros(n, dtype=float),
        "carry_pnl": np.zeros(n, dtype=float),
        "pnl": np.zeros(n, dtype=float),
    }

# Day 0 (size from START_NAV)
dcvt0 = (nav[0] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)
for inst, df in aligned.items():
    sc = scaffold[inst]
    fc0 = pd.to_numeric(df["forecast_combined"], errors="coerce").iloc[0]
    ivv0 = sc["ivv_usd"].iloc[0]
    vs0  = 0.0 if (ivv0 == 0 or np.isnan(ivv0) or np.isnan(fc0)) else dcvt0 / ivv0
    sp0  = 0.0 if np.isnan(fc0) else (fc0 * vs0) / 10.0
    tgt0 = int(np.round(sp0 * INSTRUMENTS[inst]["weight"] * PDM))

    st = state[inst]
    st["dcvt"][0] = dcvt0
    st["vol_scaler"][0] = vs0
    st["subsys_pos"][0] = sp0
    st["target"][0] = tgt0
    st["pos"][0] = tgt0
    st["trades"][0] = tgt0
    # no P&L on day 0

# Days 1..n-1
for i in range(1, n):
    dcvt = (nav[i-1] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)
    total_pnl_today = 0.0
    for inst, df in aligned.items():
        sc = scaffold[inst]
        st = state[inst]
        w  = INSTRUMENTS[inst]["weight"]

        ivv = sc["ivv_usd"].iloc[i]
        fc  = pd.to_numeric(df["forecast_combined"], errors="coerce").iloc[i]
        vs  = 0.0 if (ivv == 0 or np.isnan(ivv) or np.isnan(fc)) else dcvt / ivv
        sp  = 0.0 if np.isnan(fc) else (fc * vs) / 10.0

        tgt_unrounded = sp * w * PDM
        tgt = int(np.round(tgt_unrounded))

        prev_pos = st["pos"][i-1]
        trades   = tgt - prev_pos
        pos      = prev_pos + trades

        dp       = (sc["px"].iloc[i] - sc["px"].iloc[i-1])
        pv       = sc["point_value"]
        fx_i     = sc["fx"].iloc[i]
        carry_pnl = prev_pos * dp * pv * fx_i
        trade_pnl = 0.0  # add costs if desired: e.g., -abs(trades)*cost_per_contract

        pnl = carry_pnl + trade_pnl
        total_pnl_today += pnl

        st["dcvt"][i]        = dcvt
        st["vol_scaler"][i]  = vs
        st["subsys_pos"][i]  = sp
        st["target"][i]      = tgt
        st["pos"][i]         = pos
        st["trades"][i]      = trades
        st["carry_pnl"][i]   = carry_pnl
        st["trade_pnl"][i]   = trade_pnl
        st["pnl"][i]         = pnl

    nav[i] = nav[i-1] + total_pnl_today

# ================== SAVE PER-INSTRUMENT & PORTFOLIO SUMMARY ==================
portfolio_pnl = np.zeros(n)
for inst in aligned:
    portfolio_pnl += state[inst]["pnl"]

port_sharpe = ann_sharpe(portfolio_pnl)
port_sort   = sortino(portfolio_pnl)
port_calmar, port_mdd, port_annret = calmar_from_pnl(portfolio_pnl)
port_annvol = portfolio_pnl.std() * np.sqrt(TRADING_DAYS)
hit_rate    = (portfolio_pnl > 0).mean()

print(f"\n=== PORTFOLIO (weights={ {k: INSTRUMENTS[k]['weight'] for k in INSTRUMENTS} }) ===")
print(f"PDM used:          {PDM:.3f}")
print(f"Sharpe (ann):      {port_sharpe:.3f}")
print(f"Sortino (ann):     {port_sort:.3f}")
print(f"Ann Return (USD):  {port_annret:,.0f}")
print(f"Ann Vol (USD):     {port_annvol:,.0f}")
print(f"Max Drawdown (USD):{port_mdd:,.0f}")
print(f"Calmar:            {port_calmar:.3f}")
print(f"Hit rate:          {hit_rate:.1%}")

# Per-instrument CSVs
for inst, df in aligned.items():
    sc = scaffold[inst]
    st = state[inst]
    out = pd.DataFrame({
        "Date": df["Date"],
        "PX_CLOSE_1D": sc["px"],
        "FX_TO_USD": sc["fx"],
        "POINT_VALUE": sc["point_value"],
        "price_volatility": sc["price_vol"],
        "one_pct_move": sc["one_pct_move"],
        "block_value": sc["block_value"],
        "icv_local": sc["icv_local"],
        "ivv_usd": sc["ivv_usd"],
        "forecast_combined": pd.to_numeric(df["forecast_combined"], errors="coerce"),
        "daily_cash_vol_target": st["dcvt"],
        "volatility_scaler": st["vol_scaler"],
        "subsystem_position": st["subsys_pos"],
        "target_contracts": st["target"],
        "current_position": st["pos"],
        "trades": st["trades"],
        "trade_pnl_usd": st["trade_pnl"],
        "carry_pnl_usd": st["carry_pnl"],
        "pnl_usd": st["pnl"],
        "portfolio_nav_usd": nav,   # shared NAV
    })
    out_path = os.path.join(OUT_DIR, f"{inst}_FINAL_timeseries.csv")
    out.to_csv(out_path, index=False)
    print(f"✅ Saved {inst} → {out_path}")

# Portfolio summary CSV
summary = pd.DataFrame([{
    "pdm": PDM,
    "weights": str({k: INSTRUMENTS[k]['weight'] for k in INSTRUMENTS}),
    "sharpe": port_sharpe,
    "sortino": port_sort,
    "ann_return_usd": port_annret,
    "ann_vol_usd": port_annvol,
    "max_drawdown_usd": port_mdd,
    "calmar": port_calmar,
    "hit_rate": hit_rate,
    "obs": int(pd.Series(portfolio_pnl).dropna().shape[0]),
}])
summary_path = os.path.join(OUT_DIR, "PORTFOLIO_FINAL_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n📄 Saved portfolio summary → {summary_path}")
