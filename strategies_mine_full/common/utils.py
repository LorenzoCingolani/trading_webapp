
import os
import numpy as np
import pandas as pd

TRADING_DAYS       = 256
ANNUAL_TARGET_VOL  = 0.20
START_NAV          = 10_000_000.0
CAP                = 20.0
VAR_LOOKBACK       = 36
ALPHA              = 2.0 / (VAR_LOOKBACK + 1.0)

def pick_col(df, candidates):
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

def get_standard_cost(df: pd.DataFrame) -> float:
    cand = ["STANDARD_COST", "standard_cost", "Standard Cost", "standard cost", "STANDARD COST", "StandardCost"]
    for c in df.columns:
        if c.strip() in cand:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v): return float(v.iloc[0])
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
    print(f"[FX] No FX column found for {inst_code}. Using 1.0 fallback.")
    return pd.Series(1.0, index=df.index, dtype=float)

def get_input_price_stdev(df: pd.DataFrame) -> pd.Series:
    # absolute price stdev from input file (Excel column 'st_dev')
    candidates = ["st_dev","ST_DEV","stdev","StdDev","STD_DEV"]
    for c in df.columns:
        if c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any(): return s.ffill().bfill()
    raise KeyError("Input file is missing a usable 'st_dev' column.")

def ewma_stdev_from_net_returns(px: pd.Series, alpha: float) -> pd.Series:
    diff = px.diff().to_numpy()
    n = len(diff)
    var = np.full(n, np.nan)
    mask = ~np.isnan(diff)
    if not mask.any(): return pd.Series(np.sqrt(var), index=px.index)
    i0 = np.argmax(mask)
    var[i0] = diff[i0] ** 2
    prev = var[i0]
    for i in range(i0 + 1, n):
        x = 0.0 if np.isnan(diff[i]) else diff[i]
        prev = alpha * (x * x) + (1 - alpha) * prev
        var[i] = prev
    return pd.Series(np.sqrt(var), index=px.index)

def ewma_var_from_squared(sq: pd.Series, alpha: float) -> pd.Series:
    arr = sq.to_numpy()
    n   = len(arr)
    out = np.full(n, np.nan)
    mask = ~np.isnan(arr)
    if not mask.any(): return pd.Series(out, index=sq.index)
    i0 = np.argmax(mask)
    out[i0] = arr[i0]
    prev = out[i0]
    for i in range(i0 + 1, n):
        x2 = 0.0 if np.isnan(arr[i]) else arr[i]
        prev = alpha * x2 + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=sq.index)

def raw_signal_sharpe(forecast: pd.Series, px: pd.Series):
    ret = px.pct_change().shift(-1)     # next-day return
    pnl_raw = forecast * ret
    s = pnl_raw.dropna()
    if len(s) < 3: return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu/sd)*np.sqrt(TRADING_DAYS)

def compute_price_volatility_from_input(stdev_input: pd.Series, price_series: pd.Series) -> pd.Series:
    # price_vol% = ROUND(st_dev / price * 100, 2)
    return ((stdev_input / price_series) * 100.0).round(2)

def compute_turnover_exact_cols(df: pd.DataFrame):
    tmp = df.copy()
    tc = pd.to_numeric(tmp.get("target_contracts", pd.Series(index=tmp.index, dtype=float)),
                       errors="coerce").fillna(0.0)
    target_rounded = np.rint(tc).astype(int)
    current_pos    = np.roll(target_rounded, 1); current_pos[0] = 0
    trades_needed  = target_rounded - current_pos

    n = len(tmp); years = n / TRADING_DAYS if n else np.nan
    avg_yearly_lots = (np.nansum(np.abs(trades_needed)) / years) if (years and years>0) else np.nan
    avg_abs_pos     = np.nanmean(np.abs(current_pos)) if n else np.nan
    turnover = np.nan
    if avg_abs_pos not in (0, None) and not np.isnan(avg_abs_pos):
        turnover = avg_yearly_lots / (2.0 * avg_abs_pos)

    extra = pd.DataFrame({
        "Target_Pos_rounded": target_rounded,
        "Current_pos": current_pos,
        "trades_needed": trades_needed,
    }, index=tmp.index)
    return turnover, avg_yearly_lots, avg_abs_pos, extra

def simulate_with_nav_lag(df_base: pd.DataFrame, forecast: pd.Series,
                          px: pd.Series, fx: pd.Series,
                          ivv_usd: pd.Series, point_value: float) -> pd.DataFrame:
    """
    Sizing uses NAV(t-1).
    trade_pnl_usd on new trades at t vs exec price (EXEC_PRICE col if present, else previous close).
    carry_pnl_usd on carried pos close-to-close.
    """
    n = len(df_base)
    out = df_base.copy()
    exec_series = pd.to_numeric(out["EXEC_PRICE"], errors="coerce") if "EXEC_PRICE" in out.columns else px.shift(1)

    nav = np.zeros(n); dcvt = np.zeros(n); vol_scaler = np.zeros(n); subsystem_pos = np.zeros(n)
    target = np.zeros(n, dtype=int); pos = np.zeros(n, dtype=float); trades = np.zeros(n, dtype=float)
    trade_pnl = np.zeros(n, dtype=float); carry_pnl = np.zeros(n, dtype=float); total_pnl = np.zeros(n, dtype=float)
    exec_px = np.full(n, np.nan)

    nav[0]  = START_NAV
    dcvt[0] = (nav[0] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)

    ivv0 = ivv_usd.iloc[0]; f0 = forecast.iloc[0]
    if np.isnan(ivv0) or ivv0 == 0 or np.isnan(f0):
        vol_scaler[0] = 0.0; subsystem_pos[0] = 0.0
    else:
        vol_scaler[0] = dcvt[0] / ivv0
        subsystem_pos[0] = (f0 * vol_scaler[0]) / 10.0

    target[0] = int(np.round(subsystem_pos[0])); pos[0] = target[0]; exec_px[0] = np.nan

    for i in range(1, n):
        dcvt[i] = (nav[i-1] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)

        ivv_i, f_i = ivv_usd.iloc[i], forecast.iloc[i]
        if np.isnan(ivv_i) or ivv_i == 0 or np.isnan(f_i):
            vol_scaler[i] = 0.0; subsystem_pos[i] = 0.0
        else:
            vol_scaler[i] = dcvt[i] / ivv_i
            subsystem_pos[i] = (f_i * vol_scaler[i]) / 10.0

        target[i] = int(np.round(subsystem_pos[i]))
        trades[i] = target[i] - pos[i-1]
        pos[i]    = pos[i-1] + trades[i]

        exec_px[i] = exec_series.iloc[i]
        fx_i = fx.iloc[i] if not np.isnan(fx.iloc[i]) else 1.0

        dp_trade  = px.iloc[i] - exec_px[i]
        trade_pnl[i] = trades[i] * dp_trade * point_value * fx_i

        dp_carry  = px.iloc[i] - px.iloc[i-1]
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

def strategy_metrics_from_usd_pnl(df):
    s = df["pnl_usd"]
    sh = ann_sharpe(s)
    so = sortino(s)
    calmar, mdd, ann_ret = calmar_from_pnl(s)
    ann_vol = s.std() * np.sqrt(TRADING_DAYS)
    hit = (s > 0).mean()
    # simple proxy; usually overwritten by exact turnover
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
