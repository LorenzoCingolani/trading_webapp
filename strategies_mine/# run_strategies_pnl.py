# run_strategies_pnl.py
import os
import numpy as np
import pandas as pd

# ================== CONFIG ==================
BASE      = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
IN_DIR    = os.path.join(BASE, "all_input_files")
OUT_DIR   = os.path.join(BASE, "all_output_files")

INSTRUMENTS = {
    "RX1_small.csv": {"code": "RX1", "carry_distance": 3/12},  # quarterly
    "AD1_small.csv": {"code": "AD1", "carry_distance": 1/12},  # monthly
}

TRADING_DAYS       = 256
ANNUAL_TARGET_VOL  = 0.20
START_NAV          = 10_000_000.0
CAP                = 20.0

# Variance lookback for EWMA of squared returns (Carver)
VAR_LOOKBACK = 36
ALPHA        = 2.0 / (VAR_LOOKBACK + 1.0)

# EWMA crosses (slow = 4×fast) and Carver scalers
EWMA_SCALERS = {
    2: 10.6,
    4: 7.5,
    8: 5.3,
    16: 3.75,
    32: 2.65,
    64: 1.87,
}

CARRY_SCALER = 30.0

os.makedirs(OUT_DIR, exist_ok=True)

# ================== HELPERS ==================
def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_date_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def get_standard_cost(df: pd.DataFrame):
    cand = ["STANDARD_COST", "standard_cost", "Standard Cost", "standard cost",
            "STANDARD COST", "StandardCost"]
    for c in df.columns:
        if c.strip() in cand:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v):
                return float(v.iloc[0])
    return 0.0

def get_fx_series(df: pd.DataFrame, inst_code: str) -> pd.Series:
    exact = [
        "FX_TO_USD","fx_to_usd","FX","fx",
        "EXCHANGE_RATE_TO_USD","exchange_rate_to_usd",
        "Exchange rate","exchange rate","EXCHANGE RATE"
    ]
    for col in exact:
        if col in df.columns:
            fx = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
            if fx.notna().any():
                print(f"[FX] Using column '{col}' for {inst_code}")
                return fx
    # loose fallback
    for col in df.columns:
        key = str(col).lower().replace(" ", "").replace("_","").replace("-","")
        if (("fx" in key) or ("exchangerate" in key)) and ("usd" in key):
            fx = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
            print(f"[FX] Using loosely matched column '{col}' for {inst_code}")
            return fx
    print(f"[FX] No FX column found for {inst_code}. Using 1.0 fallback.")
    return pd.Series(1.0, index=df.index, dtype=float)

def get_input_price_stdev(df: pd.DataFrame) -> pd.Series:
    candidates = ["st_dev","ST_DEV","stdev","StdDev","STD_DEV"]
    for c in df.columns:
        if c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s.ffill().bfill()
    raise KeyError("Input file is missing a usable 'st_dev' column.")

# --- stats helpers ---
def ewma_stdev_from_net_returns(px: pd.Series, alpha: float) -> pd.Series:
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

def ewma_var_from_squared(sq: pd.Series, alpha: float) -> pd.Series:
    arr = sq.to_numpy()
    n   = len(arr)
    out = np.full(n, np.nan)
    mask = ~np.isnan(arr)
    if not mask.any():
        return pd.Series(out, index=sq.index)
    i0 = np.argmax(mask)
    out[i0] = arr[i0]
    prev = out[i0]
    for i in range(i0 + 1, n):
        x2 = 0.0 if np.isnan(arr[i]) else arr[i]
        prev = ALPHA * x2 + (1 - ALPHA) * prev
        out[i] = prev
    return pd.Series(out, index=sq.index)

def ann_sharpe(x):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu/sd)*np.sqrt(TRADING_DAYS)

def sortino(x):
    s = pd.Series(x).dropna()
    if len(s) < 3: return np.nan
    downside = s[s<0]
    dd = downside.std()
    mu = s.mean()
    return np.nan if (dd == 0 or np.isnan(dd)) else (mu/dd)*np.sqrt(TRADING_DAYS)

def max_drawdown_from_cum(cum):
    s = pd.Series(cum).fillna(0.0)
    peak = s.cummax()
    dd = s - peak
    return dd.min()

def calmar_from_pnl(raw_pnl):
    cum = raw_pnl.cumsum()
    mdd = max_drawdown_from_cum(cum)
    ann_ret = raw_pnl.mean() * TRADING_DAYS
    return (ann_ret / abs(mdd)) if (mdd not in (0, np.nan, None)) else np.nan, mdd, ann_ret

def raw_signal_sharpe(forecast: pd.Series, px: pd.Series):
    ret = px.pct_change().shift(-1)     # next-day return
    signal = forecast
    pnl_raw = signal * ret
    s = pnl_raw.dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu/sd)*np.sqrt(TRADING_DAYS)

def compute_price_volatility_from_input(stdev_input: pd.Series, price_series: pd.Series) -> pd.Series:
    # price_vol% = ROUND(st_dev / price * 100, 2)
    return ((stdev_input / price_series) * 100.0).round(2)

# --- Carver-exact turnover ---
def compute_turnover_exact_cols(df: pd.DataFrame):
    """
    turnover = avg yearly lots traded / (2 * avg abs current pos)
    using rounded targets and lagged current position.
    """
    tmp = df.copy()
    tc = pd.to_numeric(tmp.get("target_contracts", pd.Series(index=tmp.index, dtype=float)),
                       errors="coerce").fillna(0.0)
    target_rounded = np.rint(tc).astype(int)
    current_pos    = np.roll(target_rounded, 1)
    current_pos[0] = 0
    trades_needed  = target_rounded - current_pos

    n      = len(tmp)
    years  = n / TRADING_DAYS if n else np.nan
    avg_yearly_lots = (np.nansum(np.abs(trades_needed)) / years) if (years and years > 0) else np.nan
    avg_abs_pos     = np.nanmean(np.abs(current_pos)) if n else np.nan
    turnover = np.nan
    if avg_abs_pos not in (0, None) and not np.isnan(avg_abs_pos):
        turnover = avg_yearly_lots / (2.0 * avg_abs_pos)

    extra_cols = pd.DataFrame({
        "Target_Pos_rounded": target_rounded,
        "Current_pos": current_pos,
        "trades_needed": trades_needed,
    }, index=tmp.index)
    return turnover, avg_yearly_lots, avg_abs_pos, extra_cols

# --- core simulator with trade vs carry P&L split ---
def simulate_with_nav_lag(
    df_base: pd.DataFrame,
    forecast: pd.Series,
    px: pd.Series,
    fx: pd.Series,
    ivv_usd: pd.Series,
    point_value: float,
) -> pd.DataFrame:
    """
    Sizing uses NAV(t-1). P&L split:
      - trade_pnl_usd on new trades at t: (px[t] - exec_price[t]) * trades[t] * PV * FX[t]
          exec_price = df_base['EXEC_PRICE'] if present, else px[t-1]
      - carry_pnl_usd on position carried from t-1: (px[t] - px[t-1]) * pos[t-1] * PV * FX[t]
      - pnl_usd = trade_pnl_usd + carry_pnl_usd
      - nav_usd updates off pnl_usd
    """
    n = len(df_base)
    out = df_base.copy()

    # execution price series
    exec_series = pd.to_numeric(out["EXEC_PRICE"], errors="coerce") if "EXEC_PRICE" in out.columns else px.shift(1)

    nav = np.zeros(n)
    dcvt = np.zeros(n)
    vol_scaler = np.zeros(n)
    subsystem_pos = np.zeros(n)
    target = np.zeros(n, dtype=int)
    pos = np.zeros(n, dtype=float)
    trades = np.zeros(n, dtype=float)
    trade_pnl = np.zeros(n, dtype=float)
    carry_pnl = np.zeros(n, dtype=float)
    total_pnl = np.zeros(n, dtype=float)
    exec_px = np.full(n, np.nan)

    # day 0
    nav[0]  = START_NAV
    dcvt[0] = (nav[0] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)

    ivv0 = ivv_usd.iloc[0]
    f0   = forecast.iloc[0]
    if np.isnan(ivv0) or ivv0 == 0 or np.isnan(f0):
        vol_scaler[0]    = 0.0
        subsystem_pos[0] = 0.0
    else:
        vol_scaler[0]    = dcvt[0] / ivv0
        subsystem_pos[0] = (f0 * vol_scaler[0]) / 10.0

    target[0] = int(np.round(subsystem_pos[0]))
    pos[0]    = target[0]
    exec_px[0] = np.nan  # no trade P&L day 0

    # days 1..n-1
    for i in range(1, n):
        dcvt[i] = (nav[i-1] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)

        ivv_i, f_i = ivv_usd.iloc[i], forecast.iloc[i]
        if np.isnan(ivv_i) or ivv_i == 0 or np.isnan(f_i):
            vol_scaler[i]    = 0.0
            subsystem_pos[i] = 0.0
        else:
            vol_scaler[i]    = dcvt[i] / ivv_i
            subsystem_pos[i] = (f_i * vol_scaler[i]) / 10.0

        target[i] = int(np.round(subsystem_pos[i]))
        trades[i] = target[i] - pos[i-1]
        pos[i]    = pos[i-1] + trades[i]

        exec_px[i] = exec_series.iloc[i]
        fx_i = fx.iloc[i] if not np.isnan(fx.iloc[i]) else 1.0

        # trade pnl (new lots at t vs today's close)
        dp_trade = px.iloc[i] - exec_px[i]
        trade_pnl[i] = trades[i] * dp_trade * point_value * fx_i

        # carry pnl (yesterday's position close-to-close)
        dp_carry = px.iloc[i] - px.iloc[i-1]
        carry_pnl[i] = pos[i-1] * dp_carry * point_value * fx_i

        total_pnl[i] = trade_pnl[i] + carry_pnl[i]
        nav[i] = nav[i-1] + total_pnl[i]

    out["daily_cash_vol_target"] = dcvt
    out["volatility_scaler"]     = vol_scaler
    out["subsystem_position"]    = subsystem_pos
    out["target_contracts"]      = target
    out["current_position"]      = pos
    out["trades"]                = trades

    out["exec_price"]            = exec_px
    out["trade_pnl_usd"]         = trade_pnl
    out["carry_pnl_usd"]         = carry_pnl
    out["pnl_usd"]               = total_pnl
    out["nav_usd"]               = nav
    return out

def strategy_metrics_from_usd_pnl(df):
    s = df["pnl_usd"]
    sh = ann_sharpe(s)
    so = sortino(s)
    calmar, mdd, ann_ret = calmar_from_pnl(s)
    ann_vol = s.std() * np.sqrt(TRADING_DAYS)
    hit = (s > 0).mean()
    # (we still compute a simple turnover proxy, but we overwrite it with Carver-exact below)
    avg_yearly_lots = df["trades"].abs().sum() / (len(df)/TRADING_DAYS) if len(df) else np.nan
    avg_abs_pos     = df["current_position"].abs().mean()
    turnover = np.nan
    if avg_abs_pos not in (0, None) and not np.isnan(avg_abs_pos):
        turnover = avg_yearly_lots / (2.0 * avg_abs_pos)
    return {
        "sharpe": sh, "sortino": so, "ann_return_usd": ann_ret,
        "ann_vol_usd": ann_vol, "max_drawdown_usd": mdd,
        "calmar": calmar, "hit_rate": hit,
        "turnover_lots": turnover, "avg_yearly_lots": avg_yearly_lots,
        "avg_abs_pos": avg_abs_pos, "obs": int(s.dropna().shape[0]),
    }

# ================== STRATEGIES ==================
def run_ewma(df_raw: pd.DataFrame, inst_code: str):
    df = ensure_date_sorted(df_raw)

    # detect columns
    px_col  = pick_col(df, ["PX_CLOSE_1D","px_close_1d","Close","close"])
    point_val_col = pick_col(df, ["POINT_VALUE","point_value","PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE","tickValue","tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE","tickSize","tick_size"])
    if px_col is None:
        raise KeyError(f"{inst_code}: price column not found.")
    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE).")

    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    fx = get_fx_series(df, inst_code)

    # Point value
    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val  = float(pd.to_numeric(df[tick_val_col],  errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # signal stdev (for EWMA vol-adjust)
    stdev_signal = ewma_stdev_from_net_returns(px, ALPHA)

    # sizing stdev from INPUT FILE: st_dev → price_vol% → ICV → IVV
    st_dev_input = get_input_price_stdev(df)
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value
    icv_local    = block_value * price_vol                 # no /100 per your convention
    ivv_usd      = icv_local * fx

    std_cost = get_standard_cost(df_raw)
    print(f"[{inst_code}] standard_cost={std_cost:.6f}")

    for fast, scaler in EWMA_SCALERS.items():
        slow = fast * 4
        ewf = px.ewm(span=fast, adjust=False).mean()
        ews = px.ewm(span=slow, adjust=False).mean()
        raw_cross = ewf - ews
        vol_adj   = raw_cross / stdev_signal.replace(0.0, np.nan)
        forecast  = (vol_adj * scaler).clip(-CAP, CAP)

        base = pd.DataFrame({
            "Date": df["Date"] if "Date" in df.columns else pd.NaT,
            "PX_CLOSE_1D": px,
            "FX_TO_USD": fx,
            "stdev_signal": stdev_signal,
            "st_dev_input": st_dev_input,
            "price_volatility": price_vol,
            "one_pct_move": one_pct_move,
            "POINT_VALUE": point_value,
            "block_value": block_value,
            "icv_local": icv_local,
            "ivv_usd": ivv_usd,
            "forecast": forecast,
        })

        out = simulate_with_nav_lag(
            base, forecast=forecast, px=px, fx=fx, ivv_usd=ivv_usd, point_value=point_value
        )

        # Carver-exact turnover
        t_over, ayl, aap, extra_cols = compute_turnover_exact_cols(out)
        out = pd.concat([out, extra_cols], axis=1)

        # Metrics on realized PnL
        m = strategy_metrics_from_usd_pnl(out)
        m["turnover_lots"]   = t_over
        m["avg_yearly_lots"] = ayl
        m["avg_abs_pos"]     = aap

        # Raw (signal) Sharpe
        raw_sh = raw_signal_sharpe(forecast, px)

        label  = f"{inst_code}_EWMA_{fast}d_{slow}d"
        ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
        mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")

        out.to_csv(ts_csv, index=False)
        pd.DataFrame([{"instrument":inst_code,"strategy":f"EWMA_{fast}/{slow}",
                       "raw_sharpe": raw_sh, "standard_cost": std_cost, **m}]).to_csv(mt_csv, index=False)

        print(f"[{label}] RawSharpe={raw_sh:.3f} | Sharpe={m['sharpe']:.3f} | "
              f"Turnover={m['turnover_lots']:.2f} | yearly lots={m['avg_yearly_lots']:.2f} | "
              f"avg|pos|={m['avg_abs_pos']:.2f} | Obs={m['obs']}")

def run_carry(df_raw: pd.DataFrame, inst_code: str, distance: float):
    df = ensure_date_sorted(df_raw)

    px_col   = pick_col(df, ["PX_CLOSE_1D","px_close_1d","Close","close"])
    near_col = pick_col(df, ["near","NEAR"])
    far_col  = pick_col(df, ["far","FAR"])
    point_val_col = pick_col(df, ["POINT_VALUE","point_value","PointValue"])
    tick_val_col  = pick_col(df, ["TICK_VALUE","tickValue","tick_value"])
    tick_size_col = pick_col(df, ["TICK_SIZE","tickSize","tick_size"])

    if None in (px_col, near_col, far_col):
        raise KeyError(f"{inst_code}: need PX_CLOSE_1D/near/far columns.")
    if not point_val_col and not (tick_val_col and tick_size_col):
        raise KeyError(f"{inst_code}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE).")

    px   = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    near = pd.to_numeric(df[near_col], errors="coerce").astype(float)
    far  = pd.to_numeric(df[far_col],  errors="coerce").astype(float)
    fx   = get_fx_series(df, inst_code)

    # Point value
    if point_val_col:
        point_value = float(pd.to_numeric(df[point_val_col], errors="coerce").dropna().iloc[0])
    else:
        tick_val  = float(pd.to_numeric(df[tick_val_col],  errors="coerce").dropna().iloc[0])
        tick_size = float(pd.to_numeric(df[tick_size_col], errors="coerce").dropna().iloc[0])
        point_value = tick_val / tick_size

    # Carry raw forecast
    ret_near = near.diff()
    sq       = ret_near**2
    var      = ewma_var_from_squared(sq, ALPHA)
    ann_std  = np.sqrt(var) * np.sqrt(TRADING_DAYS)

    price_difference = (far - near)
    net_expected     = price_difference / float(distance)
    raw_carry        = net_expected / pd.Series(ann_std, index=df.index).replace(0.0, np.nan)
    forecast         = (raw_carry * CARRY_SCALER).clip(-CAP, CAP)

    # Sizing via input st_dev
    st_dev_input = get_input_price_stdev(df)
    price_vol    = compute_price_volatility_from_input(st_dev_input, px)
    one_pct_move = px * 0.01
    block_value  = one_pct_move * point_value
    icv_local    = block_value * price_vol
    ivv_usd      = icv_local * fx

    base = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.NaT,
        "PX_CLOSE_1D": px,
        "near": near, "far": far,
        "FX_TO_USD": fx,
        "st_dev_input": st_dev_input,
        "price_volatility": price_vol,
        "one_pct_move": one_pct_move,
        "POINT_VALUE": point_value,
        "block_value": block_value,
        "icv_local": icv_local,
        "ivv_usd": ivv_usd,
        "forecast": forecast,
    })

    out = simulate_with_nav_lag(
        base, forecast=forecast, px=px, fx=fx, ivv_usd=ivv_usd, point_value=point_value
    )

    # Raw Sharpe (signal)
    raw_sh = raw_signal_sharpe(forecast, px)

    # Metrics on realized USD PnL
    m = strategy_metrics_from_usd_pnl(out)

    # Carver-exact turnover columns
    t_over, ayl, aap, extra_cols = compute_turnover_exact_cols(out)
    out = pd.concat([out, extra_cols], axis=1)
    m["turnover_lots"]   = t_over
    m["avg_yearly_lots"] = ayl
    m["avg_abs_pos"]     = aap

    label  = f"{inst_code}_CARRY"
    ts_csv = os.path.join(OUT_DIR, f"{label}_timeseries.csv")
    mt_csv = os.path.join(OUT_DIR, f"{label}_metrics.csv")
    out.to_csv(ts_csv, index=False)
    pd.DataFrame([{"instrument":inst_code,"strategy":"CARRY","raw_sharpe":raw_sh, **m}]).to_csv(mt_csv, index=False)

    print(f"[{label}] RawSharpe={raw_sh:.3f} | Sharpe={m['sharpe']:.3f} | "
          f"Sortino={m['sortino']:.3f} | MaxDD=${m['max_drawdown_usd']:,.0f} | "
          f"Calmar={m['calmar']:.3f} | Turnover={m['turnover_lots']:.2f} | "
          f"yearly lots={m['avg_yearly_lots']:.2f} | avg|pos|={m['avg_abs_pos']:.2f} | Obs={m['obs']}")
    return [(label, ts_csv, mt_csv, m)]

# ================== MAIN ==================
if __name__ == "__main__":
    for fname, cfg in INSTRUMENTS.items():
        in_path = os.path.join(IN_DIR, fname)
        if not os.path.exists(in_path):
            print(f"❌ Missing: {in_path}")
            continue
        inst_code = cfg["code"]
        print(f"\n================= {inst_code} =================")
        raw = pd.read_csv(in_path)

        # EWMA (all crosses)
        run_ewma(raw, inst_code)

        # Carry
        run_carry(raw, inst_code, distance=cfg["carry_distance"])

    print(f"\n✅ Done. Timeseries & metrics saved under: {OUT_DIR}")
