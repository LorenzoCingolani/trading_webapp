import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\loci_\Desktop\trading_webapp"
INPUT_FILE = os.path.join(DATA_DIR, "dashboard_input.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "trading_signals.csv")


def safe_pct_change(series: pd.Series, periods: int = 5) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def add_feature_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    change_cols = [
        "brent",
        "brent_m1_m2",
        "ovx",
        "bdti",
        "tanker_index",
        "gold",
        "vix",
        "hormuz_transits",
        "risk_score",
    ]

    for col in change_cols:
        if col in out.columns:
            out[f"{col}_chg_5d"] = safe_pct_change(out[col], 5)

    if "us_crude_inventory_change" in out.columns:
        out["us_crude_inventory_change_chg_5d"] = (
            out["us_crude_inventory_change"] - out["us_crude_inventory_change"].shift(5)
        )

    return out


def add_forward_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Forward returns / changes
    out["target_brent_fwd_5d"] = out["brent"].shift(-5) / out["brent"] - 1.0
    out["target_tanker_fwd_5d"] = out["tanker_index"].shift(-5) / out["tanker_index"] - 1.0
    out["target_gold_fwd_5d"] = out["gold"].shift(-5) / out["gold"] - 1.0

    # Forward change for spread-like variables
    out["target_curve_fwd_5d"] = out["brent_m1_m2"].shift(-5) - out["brent_m1_m2"]
    out["target_ovx_fwd_5d"] = out["ovx"].shift(-5) - out["ovx"]
    out["target_vix_fwd_5d"] = out["vix"].shift(-5) - out["vix"]
    out["target_bdti_fwd_5d"] = out["bdti"].shift(-5) - out["bdti"]

    # Binary classification targets
    out["target_brent_up_5d"] = (out["target_brent_fwd_5d"] > 0).astype("float")
    out["target_curve_up_5d"] = (out["target_curve_fwd_5d"] > 0).astype("float")
    out["target_tanker_up_5d"] = (out["target_tanker_fwd_5d"] > 0).astype("float")

    # Remove labels at end where forward info unavailable
    tail_mask = out["brent"].shift(-5).isna()
    target_cols = [c for c in out.columns if c.startswith("target_")]
    out.loc[tail_mask, target_cols] = np.nan

    return out


def add_top_driver(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    contrib_cols = {
        "Brent": "contrib_brent",
        "Brent curve": "contrib_brent_curve",
        "OVX": "contrib_ovx",
        "BDTI": "contrib_bdti",
        "Tanker": "contrib_tanker",
        "Inventory": "contrib_inventory",
        "Gold": "contrib_gold",
        "VIX": "contrib_vix",
        "Hormuz": "contrib_transits",
    }

    missing = [v for v in contrib_cols.values() if v not in out.columns]
    if missing:
        raise ValueError(f"Missing contribution columns in dashboard_input.csv: {missing}")

    contrib_frame = pd.DataFrame({k: out[v] for k, v in contrib_cols.items()})

    out["top_driver"] = contrib_frame.idxmax(axis=1)
    out["top_driver_value"] = contrib_frame.max(axis=1)

    def top3_labels(row: pd.Series) -> str:
        s = row.sort_values(ascending=False).head(3)
        return " | ".join([f"{idx}:{val:.3f}" for idx, val in s.items()])

    out["top_3_drivers"] = contrib_frame.apply(top3_labels, axis=1)

    return out


def add_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) Oil curve tightening signal
    out["signal_curve_long"] = (
        (out["risk_score"] > 70)
        & (out["risk_score_chg_5d"] > 0.10)
        & (out["top_driver"].isin(["Brent curve", "OVX", "BDTI", "Tanker"]))
    ).astype(int)

    # 2) Tanker stress continuation signal
    out["signal_tanker_long"] = (
        (out["risk_score"] > 65)
        & (out["top_driver"].isin(["BDTI", "Tanker"]))
        & (out["transit_drop_z"] > 0.5)
    ).astype(int)

    # 3) Crisis premium fade signal
    out["signal_premium_fade"] = (
        (out["risk_score"] < 45)
        & (out["risk_score_chg_5d"] < -0.05)
    ).astype(int)

    # 4) Broad geopolitical escalation flag
    out["signal_escalation"] = (
        (out["risk_score"] > 75)
        & (
            (out["contrib_brent_curve"] > 0.20)
            | (out["contrib_bdti"] > 0.15)
            | (out["contrib_ovx"] > 0.15)
        )
    ).astype(int)

    return out


def add_signal_comments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    comments = []
    for _, row in out.iterrows():
        if row.get("signal_curve_long", 0) == 1:
            comments.append("Curve tightening setup")
        elif row.get("signal_tanker_long", 0) == 1:
            comments.append("Tanker stress continuation")
        elif row.get("signal_premium_fade", 0) == 1:
            comments.append("Crisis premium fading")
        elif row.get("signal_escalation", 0) == 1:
            comments.append("Escalation regime")
        else:
            comments.append("No active rule signal")

    out["signal_comment"] = comments
    return out


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = add_feature_changes(df)
    df = add_forward_targets(df)
    df = add_top_driver(df)
    df = add_rule_signals(df)
    df = add_signal_comments(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")

    latest = df.iloc[-1]
    print("\nLatest trading snapshot")
    print("-----------------------")
    print(f"Date: {latest['date']}")
    print(f"Risk score: {latest.get('risk_score', np.nan):.1f}")
    print(f"Risk regime: {latest.get('risk_regime', 'NA')}")
    print(f"Top driver: {latest.get('top_driver', 'NA')}")
    print(f"Top 3 drivers: {latest.get('top_3_drivers', 'NA')}")
    print(f"Signal comment: {latest.get('signal_comment', 'NA')}")

    print("\nSignals")
    print("-------")
    print(f"Curve long: {int(latest.get('signal_curve_long', 0))}")
    print(f"Tanker long: {int(latest.get('signal_tanker_long', 0))}")
    print(f"Premium fade: {int(latest.get('signal_premium_fade', 0))}")
    print(f"Escalation: {int(latest.get('signal_escalation', 0))}")


if __name__ == "__main__":
    main()