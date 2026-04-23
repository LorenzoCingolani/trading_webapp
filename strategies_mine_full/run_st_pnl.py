import os
import sys
import pandas as pd

# Ensure parent project dir is on sys.path so package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE      = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
IN_DIR    = os.path.join(BASE, "all_input_files")
OUT_DIR   = os.path.join(BASE, "all_output_files")

from strategies_mine_full.strategies.carry import run_carry
from strategies_mine_full.strategies.ewma import run_ewma
#from ..common.utils import annual_sharpe


INSTRUMENTS = {
    "RX1_small.csv": {"code": "RX1", "carry_distance": 3/12},  # quarterly
    "AD1_small.csv": {"code": "AD1", "carry_distance": 1/12},  # monthly
}

os.makedirs(OUT_DIR, exist_ok=True)

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
        run_ewma(raw, inst_code, OUT_DIR)

        # Carry
        run_carry(
            raw,
            inst_code,
            distance_years=cfg["carry_distance"],  # <-- keyword name must match carry.py
            OUT_DIR=OUT_DIR                        # <-- pass your output folder
        )   



    print(f"\n✅ Done. Timeseries & metrics saved under: {OUT_DIR}")
