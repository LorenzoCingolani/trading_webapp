"""Carry strategy module."""
import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import (
    TRADING_DAYS,
    ANNUAL_TARGET_VOL,
    ALPHA,
    CARRY_SCALER,
    pick_col,
    compute_carry_forecast,
    build_sizing_scaffold,
)


class CarryStrategy:
    """Carry-based position sizing strategy."""

    def __init__(
        self,
        instruments: Dict[str, Dict],
        aligned_data: Dict[str, pd.DataFrame],
        scaffold: Dict[str, Dict],
        pdm: float,
        start_nav: float = 10_000_000.0,
        carry_distance_years: Dict[str, float] = None,
    ):
        """
        Initialize Carry strategy.

        Args:
            instruments: Config dict {inst_name -> {'weight': ..., 'input_file': ...}}
            aligned_data: {inst_name -> DataFrame with aligned dates}
            scaffold: {inst_name -> sizing scaffold dict}
            pdm: Portfolio Diversification Multiplier
            start_nav: Starting NAV
            carry_distance_years: {inst_name -> distance in years} for carry calc
        """
        self.instruments = instruments
        self.aligned_data = aligned_data
        self.scaffold = scaffold
        self.pdm = pdm
        self.start_nav = start_nav
        self.n_dates = len(next(iter(aligned_data.values())))
        self.carry_distance_years = carry_distance_years or {
            inst: 0.25 for inst in instruments
        }  # default quarterly

    def compute_positions(self) -> Tuple[Dict, np.ndarray]:
        """
        Compute positions using carry signals.

        Returns:
            (state_dict, nav_array)
            - state_dict: {inst_name -> {metric_name -> array}}
            - nav_array: portfolio NAV over time
        """
        nav = np.zeros(self.n_dates, dtype=float)
        nav[0] = self.start_nav

        state = {}
        forecasts = {}

        # Pre-compute carry forecasts for all instruments
        for inst, df in self.aligned_data.items():
            near_col = pick_col(df, ["near", "NEAR"])
            far_col = pick_col(df, ["far", "FAR"])

            if near_col and far_col:
                near = pd.to_numeric(df[near_col], errors="coerce")
                far = pd.to_numeric(df[far_col], errors="coerce")
                distance = self.carry_distance_years.get(inst, 0.25)
                forecasts[inst] = compute_carry_forecast(
                    near, far, distance, alpha=ALPHA, scaler=CARRY_SCALER, cap=20.0
                )
            else:
                # Fallback to uniform forecast if near/far not available
                forecasts[inst] = pd.Series(0.0, index=df.index)

        # Initialize state arrays
        for inst in self.aligned_data:
            state[inst] = {
                "dcvt": np.zeros(self.n_dates, dtype=float),
                "vol_scaler": np.zeros(self.n_dates, dtype=float),
                "subsys_pos": np.zeros(self.n_dates, dtype=float),
                "target": np.zeros(self.n_dates, dtype=int),
                "pos": np.zeros(self.n_dates, dtype=float),
                "trades": np.zeros(self.n_dates, dtype=float),
                "trade_pnl": np.zeros(self.n_dates, dtype=float),
                "carry_pnl": np.zeros(self.n_dates, dtype=float),
                "pnl": np.zeros(self.n_dates, dtype=float),
                "forecast": np.zeros(self.n_dates, dtype=float),
            }

        # Day 0
        dcvt0 = (nav[0] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)
        for inst, df in self.aligned_data.items():
            sc = self.scaffold[inst]
            fc0 = forecasts[inst].iloc[0]
            ivv0 = sc["ivv_usd"].iloc[0]
            vs0 = (
                0.0
                if (ivv0 == 0 or np.isnan(ivv0) or np.isnan(fc0))
                else dcvt0 / ivv0
            )
            sp0 = 0.0 if np.isnan(fc0) else (fc0 * vs0) / 10.0
            tgt0 = int(
                np.round(sp0 * self.instruments[inst]["weight"] * self.pdm)
            )

            st = state[inst]
            st["dcvt"][0] = dcvt0
            st["vol_scaler"][0] = vs0
            st["subsys_pos"][0] = sp0
            st["target"][0] = tgt0
            st["pos"][0] = tgt0
            st["trades"][0] = tgt0
            st["forecast"][0] = fc0

        # Days 1..n-1
        for i in range(1, self.n_dates):
            dcvt = (nav[i - 1] * ANNUAL_TARGET_VOL) / np.sqrt(TRADING_DAYS)
            total_pnl_today = 0.0
            for inst, df in self.aligned_data.items():
                sc = self.scaffold[inst]
                st = state[inst]
                w = self.instruments[inst]["weight"]

                ivv = sc["ivv_usd"].iloc[i]
                fc = forecasts[inst].iloc[i]
                vs = (
                    0.0
                    if (ivv == 0 or np.isnan(ivv) or np.isnan(fc))
                    else dcvt / ivv
                )
                sp = 0.0 if np.isnan(fc) else (fc * vs) / 10.0

                tgt_unrounded = sp * w * self.pdm
                tgt = int(np.round(tgt_unrounded))

                prev_pos = st["pos"][i - 1]
                trades = tgt - prev_pos
                pos = prev_pos + trades

                dp = sc["px"].iloc[i] - sc["px"].iloc[i - 1]
                pv = sc["point_value"]
                fx_i = sc["fx"].iloc[i]
                carry_pnl = prev_pos * dp * pv * fx_i
                trade_pnl = 0.0

                pnl = carry_pnl + trade_pnl
                total_pnl_today += pnl

                st["dcvt"][i] = dcvt
                st["vol_scaler"][i] = vs
                st["subsys_pos"][i] = sp
                st["target"][i] = tgt
                st["pos"][i] = pos
                st["trades"][i] = trades
                st["carry_pnl"][i] = carry_pnl
                st["trade_pnl"][i] = trade_pnl
                st["pnl"][i] = pnl
                st["forecast"][i] = fc

            nav[i] = nav[i - 1] + total_pnl_today

        return state, nav

