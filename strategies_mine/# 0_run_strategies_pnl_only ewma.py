#!/usr/bin/env python
"""Main entry point for portfolio simulation and P&L analysis."""
import os
import sys
import numpy as np
import pandas as pd

# Add strategies_mine to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.utils import (
    TRADING_DAYS,
    ANNUAL_TARGET_VOL,
    PDM_CAP,
    ALPHA,
    ensure_dates,
    pick_col,
    ewma_stdev_net,
    ann_sharpe,
    sortino,
    calmar_from_pnl,
    compute_pdm_from_prices,
    load_and_prepare_instruments,
    align_dates_across_instruments,
    build_sizing_scaffold,
)
from strategies.ewma import EWMAStrategy
from strategies.carry import CarryStrategy


# ================== CONFIG ==================
BASE = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
IN_DIR = os.path.join(BASE, "all_input_files")
OUT_DIR = os.path.join(BASE, "all_output_files")

# Instruments and portfolio weights
INSTRUMENTS = {
    "RX1": {"weight": 0.5, "input_file": "RX1_small.csv"},
    "AD1": {"weight": 0.5, "input_file": "AD1_small.csv"},
}

# Carry distances (in years) for each instrument
CARRY_DISTANCES = {
    "RX1": 0.25,  # quarterly
    "AD1": 0.25,  # quarterly
}

START_NAV = 10_000_000.0

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    """Run portfolio simulation."""
    print("\n" + "=" * 60)
    print("PORTFOLIO FRAMEWORK - MAIN EXECUTION")
    print("=" * 60)

    # ================== LOAD DATA ==================
    print("\n[1/6] Loading instrument data...")
    combo = {}
    inputs = {}

    for inst, cfg in INSTRUMENTS.items():
        comb_path = os.path.join(OUT_DIR, f"{inst}_COMBINED.csv")
        if not os.path.exists(comb_path):
            print(f"❌ Missing combined file for {inst}: {comb_path}")
            continue
        dfc = ensure_dates(pd.read_csv(comb_path))
        px_col = pick_col(dfc, ["PX_CLOSE_1D", "px_close_1d", "Close", "close"])
        if px_col != "PX_CLOSE_1D":
            dfc = dfc.rename(columns={px_col: "PX_CLOSE_1D"})
        combo[inst] = dfc[["Date", "PX_CLOSE_1D", "forecast_combined"]].copy()

        # Input file for POINT_VALUE/FX
        in_path = os.path.join(IN_DIR, cfg["input_file"])
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
        raise SystemExit("❌ No instruments loaded. Ensure *_COMBINED.csv exist.")

    print(f"✅ Loaded {len(combo)} instruments")

    # ================== ALIGN DATES ==================
    print("\n[2/6] Aligning dates across instruments...")
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

    print(f"✅ Common date range: {dates.min()} to {dates.max()} ({len(dates)} days)")

    # ================== COMPUTE PDM ==================
    print("\n[3/6] Computing PDM from price correlations...")
    aligned_px = {
        inst: pd.to_numeric(df["PX_CLOSE_1D"], errors="coerce")
        for inst, df in aligned.items()
    }
    weights_map = {inst: INSTRUMENTS[inst]["weight"] for inst in aligned}
    pdm, pdm_cols, corr_mat = compute_pdm_from_prices(
        aligned_px, weights_map, cap=PDM_CAP
    )

    print(f"✅ PDM: {pdm:.3f} (cap {PDM_CAP})")
    print(f"   Instruments: {pdm_cols}")
    print(f"   Weights: {weights_map}")
    if corr_mat is not None:
        print("\n   Correlation matrix (% returns):")
        print("   " + corr_mat.round(3).to_string().replace("\n", "\n   "))

    # ================== BUILD SIZING SCAFFOLD ==================
    print("\n[4/6] Building sizing scaffold per instrument...")
    scaffold = {}
    for inst, df in aligned.items():
        point_value = float(
            pd.to_numeric(df["POINT_VALUE"], errors="coerce").dropna().iloc[0]
        )
        scaffold[inst] = build_sizing_scaffold(inst, df, point_value)

    print(f"✅ Built scaffold for {len(scaffold)} instruments")

    # ================== RUN EWMA STRATEGY ==================
    print("\n[5a/7] Running EWMA strategy simulation...")
    ewma_strategy = EWMAStrategy(
        instruments=INSTRUMENTS,
        aligned_data=aligned,
        scaffold=scaffold,
        pdm=pdm,
        start_nav=START_NAV,
    )
    ewma_state, ewma_nav = ewma_strategy.compute_positions()
    print(f"✅ EWMA simulation complete: {len(ewma_nav)} days, final NAV ${ewma_nav[-1]:,.0f}")

    # ================== RUN CARRY STRATEGY ==================
    print("\n[5b/7] Running Carry strategy simulation...")
    carry_strategy = CarryStrategy(
        instruments=INSTRUMENTS,
        aligned_data=aligned,
        scaffold=scaffold,
        pdm=pdm,
        start_nav=START_NAV,
        carry_distance_years=CARRY_DISTANCES,
    )
    carry_state, carry_nav = carry_strategy.compute_positions()
    print(f"✅ Carry simulation complete: {len(carry_nav)} days, final NAV ${carry_nav[-1]:,.0f}")

    # ================== COMPUTE METRICS & SAVE ==================
    print("\n[6/7] Computing EWMA metrics and saving outputs...")

    portfolio_pnl = np.zeros(len(nav))
    for inst in aligned:
        portfolio_pnl += state[inst]["pnl"]

    port_sharpe = ann_sharpe(portfolio_pnl)
    port_sort = sortino(portfolio_pnl)
    port_calmar, port_mdd, port_annret = calmar_from_pnl(portfolio_pnl)
    port_annvol = portfolio_pnl.std() * np.sqrt(TRADING_DAYS)
    hit_rate = (portfolio_pnl > 0).mean()

    print("\n📊 PORTFOLIO SUMMARY")
    print("-" * 60)
    print(f"PDM:                  {pdm:.3f}")
    print(f"Sharpe (ann):         {port_sharpe:.3f}")
    print(f"Sortino (ann):        {port_sort:.3f}")
    print(f"Ann Return (USD):     ${port_annret:,.0f}")
    print(f"Ann Vol (USD):        ${port_annvol:,.0f}")
    print(f"Max Drawdown (USD):   ${port_mdd:,.0f}")
    print(f"Calmar:               {port_calmar:.3f}")
    print(f"Hit Rate:             {hit_rate:.1%}")
    print(f"Final NAV (USD):      ${nav[-1]:,.0f}")

    # Per-instrument CSVs
    for inst, df in aligned.items():
        sc = scaffold[inst]
        st = state[inst]
        out = pd.DataFrame(
            {
                "Date": df["Date"],
                "PX_CLOSE_1D": sc["px"],
                "FX_TO_USD": sc["fx"],
                "POINT_VALUE": sc["point_value"],
                "price_volatility": sc["price_vol"],
                "one_pct_move": sc["one_pct_move"],
                "block_value": sc["block_value"],
                "icv_local": sc["icv_local"],
                "ivv_usd": sc["ivv_usd"],
                "forecast_combined": pd.to_numeric(
                    df["forecast_combined"], errors="coerce"
                ),
                "daily_cash_vol_target": st["dcvt"],
                "volatility_scaler": st["vol_scaler"],
                "subsystem_position": st["subsys_pos"],
                "target_contracts": st["target"],
                "current_position": st["pos"],
                "trades": st["trades"],
                "trade_pnl_usd": st["trade_pnl"],
                "carry_pnl_usd": st["carry_pnl"],
                "pnl_usd": st["pnl"],
                "portfolio_nav_usd": nav,
            }
        )
        out_path = os.path.join(OUT_DIR, f"{inst}_FINAL_timeseries.csv")
        out.to_csv(out_path, index=False)
        print(f"✅ Saved {inst} → {out_path}")

    # Portfolio summary CSV
    summary = pd.DataFrame(
        [
            {
                "pdm": pdm,
                "weights": str({k: INSTRUMENTS[k]["weight"] for k in INSTRUMENTS}),
                "sharpe": port_sharpe,
                "sortino": port_sort,
                "ann_return_usd": port_annret,
                "ann_vol_usd": port_annvol,
                "max_drawdown_usd": port_mdd,
                "calmar": port_calmar,
                "hit_rate": hit_rate,
                "final_nav_usd": nav[-1],
                "obs": int(pd.Series(portfolio_pnl).dropna().shape[0]),
            }
        ]
    )
    summary_path = os.path.join(OUT_DIR, "PORTFOLIO_FINAL_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Saved portfolio summary → {summary_path}")

    print("\n" + "=" * 60)
    print("✅ EXECUTION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
