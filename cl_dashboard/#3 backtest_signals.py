import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\loci_\Desktop\trading_webapp"
INPUT_FILE = os.path.join(DATA_DIR, "trading_signals.csv")
OUTPUT_SUMMARY = os.path.join(DATA_DIR, "signal_backtest_summary.csv")
OUTPUT_TRADES = os.path.join(DATA_DIR, "signal_backtest_trades.csv")


def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def hit_rate(x: pd.Series, direction: str = "up") -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    if direction == "up":
        return float((x > 0).mean())
    if direction == "down":
        return float((x < 0).mean())
    raise ValueError("direction must be 'up' or 'down'")


def summarize_signal(
    df: pd.DataFrame,
    signal_col: str,
    target_cols: list[str],
    signal_name: str,
) -> list[dict]:
    rows = []
    signal_df = df[df[signal_col] == 1].copy()

    n = len(signal_df)
    if n == 0:
        for t in target_cols:
            rows.append(
                {
                    "signal": signal_name,
                    "target": t,
                    "n_obs": 0,
                    "avg_target": np.nan,
                    "median_target": np.nan,
                    "hit_rate_up": np.nan,
                    "hit_rate_down": np.nan,
                    "std_target": np.nan,
                    "t_stat": np.nan,
                }
            )
        return rows

    for t in target_cols:
        x = pd.to_numeric(signal_df[t], errors="coerce").dropna()
        n_t = len(x)

        avg = float(x.mean()) if n_t else np.nan
        med = float(x.median()) if n_t else np.nan
        std = float(x.std(ddof=1)) if n_t > 1 else np.nan
        hr_up = float((x > 0).mean()) if n_t else np.nan
        hr_down = float((x < 0).mean()) if n_t else np.nan

        if n_t > 1 and std not in [0, np.nan] and not pd.isna(std):
            t_stat = avg / (std / np.sqrt(n_t))
        else:
            t_stat = np.nan

        rows.append(
            {
                "signal": signal_name,
                "target": t,
                "n_obs": n_t,
                "avg_target": avg,
                "median_target": med,
                "hit_rate_up": hr_up,
                "hit_rate_down": hr_down,
                "std_target": std,
                "t_stat": t_stat,
            }
        )

    return rows


def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    trades = []

    signal_map = {
        "signal_curve_long": {
            "trade_type": "Curve Long",
            "target_col": "target_curve_fwd_5d",
        },
        "signal_tanker_long": {
            "trade_type": "Tanker Long",
            "target_col": "target_tanker_fwd_5d",
        },
        "signal_premium_fade": {
            "trade_type": "Premium Fade",
            "target_col": "target_brent_fwd_5d",
        },
        "signal_escalation": {
            "trade_type": "Escalation",
            "target_col": "target_brent_fwd_5d",
        },
    }

    for signal_col, meta in signal_map.items():
        active = df[df[signal_col] == 1].copy()
        if active.empty:
            continue

        for _, row in active.iterrows():
            trades.append(
                {
                    "date": row["date"],
                    "signal": signal_col,
                    "trade_type": meta["trade_type"],
                    "risk_score": row.get("risk_score", np.nan),
                    "risk_regime": row.get("risk_regime", np.nan),
                    "top_driver": row.get("top_driver", np.nan),
                    "top_3_drivers": row.get("top_3_drivers", np.nan),
                    "signal_comment": row.get("signal_comment", np.nan),
                    "realized_target": row.get(meta["target_col"], np.nan),
                }
            )

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df = trade_df.sort_values(["date", "signal"]).reset_index(drop=True)
    return trade_df


def overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "signal_curve_long",
        "signal_tanker_long",
        "signal_premium_fade",
        "signal_escalation",
    ]

    out = df[["date"] + sig_cols].copy()
    out["n_active_signals"] = out[sig_cols].sum(axis=1)

    return out


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required_cols = [
        "signal_curve_long",
        "signal_tanker_long",
        "signal_premium_fade",
        "signal_escalation",
        "target_brent_fwd_5d",
        "target_curve_fwd_5d",
        "target_tanker_fwd_5d",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in trading_signals.csv: {missing}")

    summary_rows = []

    summary_rows += summarize_signal(
        df,
        signal_col="signal_curve_long",
        target_cols=["target_curve_fwd_5d", "target_brent_fwd_5d"],
        signal_name="Curve Long",
    )

    summary_rows += summarize_signal(
        df,
        signal_col="signal_tanker_long",
        target_cols=["target_tanker_fwd_5d", "target_bdti_fwd_5d", "target_brent_fwd_5d"],
        signal_name="Tanker Long",
    )

    summary_rows += summarize_signal(
        df,
        signal_col="signal_premium_fade",
        target_cols=["target_brent_fwd_5d", "target_curve_fwd_5d"],
        signal_name="Premium Fade",
    )

    summary_rows += summarize_signal(
        df,
        signal_col="signal_escalation",
        target_cols=["target_brent_fwd_5d", "target_curve_fwd_5d", "target_tanker_fwd_5d"],
        signal_name="Escalation",
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    trade_df = build_trade_log(df)
    trade_df.to_csv(OUTPUT_TRADES, index=False)

    overlap_df = overlap_summary(df)

    print(f"Saved: {OUTPUT_SUMMARY}")
    print(f"Saved: {OUTPUT_TRADES}")

    print("\nBacktest summary")
    print("----------------")
    for signal_name in summary_df["signal"].dropna().unique():
        sub = summary_df[summary_df["signal"] == signal_name]
        print(f"\n{signal_name}")
        print(sub[["target", "n_obs", "avg_target", "hit_rate_up", "t_stat"]].to_string(index=False))

    print("\nSignal overlap")
    print("--------------")
    print(overlap_df["n_active_signals"].value_counts().sort_index().to_string())

    latest = df.iloc[-1]
    print("\nLatest row")
    print("----------")
    print(f"Date: {latest['date']}")
    print(f"Risk score: {latest.get('risk_score', np.nan):.1f}")
    print(f"Risk regime: {latest.get('risk_regime', 'NA')}")
    print(f"Top driver: {latest.get('top_driver', 'NA')}")


if __name__ == "__main__":
    main()