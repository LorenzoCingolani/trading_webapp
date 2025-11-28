
import os
import pandas as pd

BASE      = r"C:\Users\loci_\Desktop\trading_webapp\DATA"
IN_DIR    = os.path.join(BASE, "all_input_files")
OUT_DIR   = os.path.join(BASE, "all_output_files")

# from .strategies.ewma import run_ewma
# from strategies.carry import run_carry

from strategies_mine_full.strategies.ewma import run_ewma
from strategies_mine_full.strategies.carry import run_carry

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

        run_ewma(raw, inst_code, OUT_DIR)
        run_carry(raw, inst_code, distance_years=cfg["carry_distance"], OUT_DIR=OUT_DIR)

    print(f"\n✅ Done. Timeseries & metrics saved under: {OUT_DIR}")
