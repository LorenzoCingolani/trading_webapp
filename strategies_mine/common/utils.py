"""Utility functions for data processing, metrics, and position sizing."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Configuration constants
TRADING_DAYS = 256
ANNUAL_TARGET_VOL = 0.20
PDM_CAP = 2.5
VAR_LOOKBACK = 36
ALPHA = 2.0 / (VAR_LOOKBACK + 1.0)  # EWMA variance decay
CARRY_SCALER = 30.0  # Carry signal multiplier


def pick_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Pick first available column from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, handle NaT, sort, and reset index."""
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
        prev = alpha * (x * x) + (1 - alpha) * prev
        var[i] = prev
    return pd.Series(np.sqrt(var), index=px.index)


def ann_sharpe(x) -> float:
    """Annualized Sharpe ratio."""
    s = pd.Series(x).dropna()
    if len(s) < 3:
        return np.nan
    mu, sd = s.mean(), s.std()
    return np.nan if (sd == 0 or np.isnan(sd)) else (mu / sd) * np.sqrt(TRADING_DAYS)


def sortino(x) -> float:
    """Sortino ratio (downside deviation)."""
    s = pd.Series(x).dropna()
    if len(s) < 3:
        return np.nan
    dd = s[s < 0].std()
    mu = s.mean()
    return np.nan if (dd == 0 or np.isnan(dd)) else (mu / dd) * np.sqrt(TRADING_DAYS)


def max_drawdown_from_cum(cum) -> float:
    """Maximum drawdown from cumulative series."""
    s = pd.Series(cum).fillna(0.0)
    peak = s.cummax()
    dd = s - peak
    return dd.min()  # negative


def calmar_from_pnl(p) -> Tuple[float, float, float]:
    """Calmar ratio, max drawdown, and annualized return from P&L series."""
    cum = p.cumsum()
    mdd = max_drawdown_from_cum(cum)
    ann_ret = p.mean() * TRADING_DAYS
    if (mdd is None) or np.isnan(mdd) or mdd == 0:
        return np.nan, mdd, ann_ret
    return ann_ret / abs(mdd), mdd, ann_ret


def compute_pdm_from_prices(
    aligned_price_series: Dict[str, pd.Series],
    weights_dict: Dict[str, float],
    cap: float = PDM_CAP
) -> Tuple[float, list, Optional[pd.DataFrame]]:
    """
    Compute Portfolio Diversification Multiplier (PDM) from price pct-change correlations.
    
    Args:
        aligned_price_series: {instrument_name -> Series of prices on common dates}
        weights_dict: {instrument_name -> portfolio weight}
        cap: Maximum PDM cap (default 2.5)
    
    Returns:
        (pdm_value, instruments_used, correlation_matrix)
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


def load_and_prepare_instruments(
    instruments: Dict[str, Dict],
    in_dir: str,
    out_dir: str
) -> Tuple[Dict, Dict, Dict]:
    """
    Load combined and input files for all instruments.
    
    Returns:
        (combo_data, inputs_data, scaffolds)
    """
    combo = {}
    inputs = {}
    
    for inst, cfg in instruments.items():
        comb_path = os.path.join(out_dir, f"{inst}_COMBINED.csv")
        if not os.path.exists(comb_path):
            print(f"❌ Missing combined file for {inst}: {comb_path}")
            continue
        dfc = ensure_dates(pd.read_csv(comb_path))
        px_col = pick_col(dfc, ["PX_CLOSE_1D", "px_close_1d", "Close", "close"])
        if px_col != "PX_CLOSE_1D":
            dfc = dfc.rename(columns={px_col: "PX_CLOSE_1D"})
        combo[inst] = dfc[["Date", "PX_CLOSE_1D", "forecast_combined"]].copy()

        # Input file for POINT_VALUE/FX
        in_path = os.path.join(in_dir, cfg["input_file"])
        dfi = ensure_dates(pd.read_csv(in_path))

        fx_col = pick_col(dfi, ["FX_TO_USD", "fx_to_usd", "FX", "fx"])
        pt_col = pick_col(dfi, ["POINT_VALUE", "point_value", "PointValue"])
        tv_col = pick_col(dfi, ["TICK_VALUE", "tickValue", "tick_value"])
        ts_col = pick_col(dfi, ["TICK_SIZE", "tickSize", "tick_size"])

        if pt_col:
            dfi["POINT_VALUE"] = pd.to_numeric(dfi[pt_col], errors="coerce")
        elif tv_col and ts_col:
            dfi["POINT_VALUE"] = (
                pd.to_numeric(dfi[tv_col], errors="coerce")
                / pd.to_numeric(dfi[ts_col], errors="coerce")
            )
        else:
            raise KeyError(
                f"{inst}: need POINT_VALUE or (TICK_VALUE & TICK_SIZE) in {cfg['input_file']}"
            )

        dfi["FX_TO_USD"] = (
            pd.to_numeric(dfi[fx_col], errors="coerce") if fx_col else 1.0
        )
        inputs[inst] = dfi[["Date", "POINT_VALUE", "FX_TO_USD"]].copy()

    if not combo:
        raise SystemExit("No instruments loaded. Ensure *_COMBINED.csv exist.")

    return combo, inputs, None


def align_dates_across_instruments(
    combo: Dict[str, pd.DataFrame],
    inputs: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Align all instruments to common date index."""
    dates = None
    for inst, df in combo.items():
        dates = (
            df["Date"]
            if dates is None
            else pd.merge(
                pd.DataFrame({"Date": dates}), df[["Date"]], on="Date", how="inner"
            )["Date"]
        )
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)

    aligned = {}
    for inst in combo:
        dfc = combo[inst]
        dfi = inputs[inst]
        m = pd.merge(
            pd.merge(pd.DataFrame({"Date": dates}), dfc, on="Date", how="left"),
            dfi,
            on="Date",
            how="left",
        ).dropna(subset=["PX_CLOSE_1D"])
        m = m.sort_values("Date").reset_index(drop=True)
        aligned[inst] = m

    return aligned


def build_sizing_scaffold(
    inst: str,
    df: pd.DataFrame,
    point_value: float
) -> Dict[str, np.ndarray]:
    """Build per-instrument sizing scaffold (vol, block value, etc.)."""
    px = pd.to_numeric(df["PX_CLOSE_1D"], errors="coerce")
    fx = pd.to_numeric(df["FX_TO_USD"], errors="coerce").fillna(1.0)

    stdev = ewma_stdev_net(px, ALPHA)
    price_vol = (stdev / px) * 100.0
    price_vol = price_vol.round(2)
    one_pct_move = px * 0.01
    block_value = one_pct_move * point_value
    icv_local = block_value * price_vol
    ivv_usd = icv_local * fx

    return {
        "px": px,
        "fx": fx,
        "point_value": point_value,
        "stdev": stdev,
        "price_vol": price_vol,
        "one_pct_move": one_pct_move,
        "block_value": block_value,
        "icv_local": icv_local,
        "ivv_usd": ivv_usd,
    }


def compute_carry_forecast(
    near: pd.Series,
    far: pd.Series,
    distance_years: float,
    alpha: float = ALPHA,
    scaler: float = CARRY_SCALER,
    cap: float = 20.0
) -> pd.Series:
    """
    Compute carry forecast from near/far price curves.
    
    Args:
        near: Near-leg price series
        far: Far-leg price series
        distance_years: Carry distance in years (e.g., 0.25 for quarterly)
        alpha: EWMA decay factor
        scaler: Signal multiplier
        cap: Forecast cap/floor
    
    Returns:
        Capped carry forecast series
    """
    # EWMA variance on near-leg returns
    ret_near = near - near.shift(1)
    sq = ret_near ** 2
    var = ewma_stdev_net(near, alpha) ** 2
    ann_std = np.sqrt(var) * np.sqrt(TRADING_DAYS)
    
    # Carry signal: expected price move / annualized stdev
    price_diff = far - near
    net_expected = price_diff / float(distance_years)
    raw_carry = net_expected / pd.Series(ann_std, index=near.index).replace(0.0, np.nan)
    
    # Scale and cap
    forecast = (raw_carry * scaler).clip(-cap, cap)
    return forecast


def compute_ewma_forecast(
    px: pd.Series,
    alpha: float = ALPHA,
    scaler: float = 1.0,
    cap: float = 20.0
) -> pd.Series:
    """
    Compute EWMA-based trend forecast from price momentum.
    
    Args:
        px: Price series
        alpha: EWMA decay factor
        scaler: Signal multiplier
        cap: Forecast cap/floor
    
    Returns:
        Capped EWMA forecast series
    """
    # EWMA volatility
    stdev = ewma_stdev_net(px, alpha)
    
    # Momentum: price return / volatility
    ret = px.pct_change() * 100.0  # Convert to basis points
    momentum = ret / (stdev.replace(0.0, np.nan) * stdev.replace(0.0, np.nan))
    
    # Scale and cap
    forecast = (momentum * scaler).clip(-cap, cap)
    return forecast


# Import os at module level for use in functions
import os
