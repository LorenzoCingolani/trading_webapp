import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from io import BytesIO


DATA_DIR = r"C:\Users\loci_\Desktop\trading_webapp"
EIA_KEY = "qRMEAFxblu0m3fS245A6PUEs4JVboJuA7qZkNzhb"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_csv(df, name):
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def zscore(series, window=60):
    mean = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std()
    return ((series - mean) / std).clip(-3, 3)


def minmax_0_100(series, window=180):
    rmin = series.rolling(window, min_periods=30).min()
    rmax = series.rolling(window, min_periods=30).max()
    return (100 * (series - rmin) / (rmax - rmin)).clip(0, 100)


# -----------------------------
# DATA SOURCES
# -----------------------------

def fetch_brent():
    url = (
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={EIA_KEY}"
        "&frequency=daily"
        "&data[0]=value"
        "&facets[series][]=RBRTE"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&length=2000"
    )

    r = requests.get(url)
    js = r.json()

    data = js["response"]["data"]
    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"])

    df = df[["date", "value"]].sort_values("date")
    return df


def fetch_yf(ticker):
    df = yf.download(ticker, period="2y", progress=False)
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["date", "value"]
    return df


def fetch_ovx():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=OVXCLS"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_vix():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_us_crude_stocks_change():
    url = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    xls = pd.ExcelFile(BytesIO(r.content), engine="xlrd")

    header_row = None
    target_sheet = None

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)

        for i in range(min(30, len(raw))):
            row_vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if "date" in row_vals:
                header_row = i
                target_sheet = sheet
                break

        if target_sheet is not None:
            break

    if target_sheet is None:
        raise RuntimeError(
            f"Could not find header row in EIA crude stocks XLS file. Sheets found: {xls.sheet_names}"
        )

    df = pd.read_excel(xls, sheet_name=target_sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = df.columns[0]
    value_col = df.columns[1]

    df = df.rename(columns={date_col: "date", value_col: "stocks"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stocks"] = pd.to_numeric(df["stocks"], errors="coerce")

    df = df[["date", "stocks"]].dropna().sort_values("date")
    df["value"] = df["stocks"].diff() / 1000.0

    return df[["date", "value"]].dropna()


# -----------------------------
# MAIN
# -----------------------------

def main():

    ensure_dir(DATA_DIR)

    brent = fetch_brent()
    tanker = fetch_yf("BWET")
    gold = fetch_yf("GC=F")

    ovx = fetch_ovx()
    vix = fetch_vix()

    inventory = fetch_us_crude_stocks_change()

    # proxies
    brent_curve = brent.copy()
    brent_curve["value"] = brent_curve["value"] - brent_curve["value"].rolling(20).mean()

    bdti = tanker.copy()

    hormuz = brent.copy()
    vol = brent["value"].pct_change().rolling(10).std()
    hormuz["value"] = (95 - vol * 500).clip(50, 110)

    # merge
    df = (
        brent.rename(columns={"value": "brent"})
        .merge(brent_curve.rename(columns={"value": "brent_m1_m2"}), on="date")
        .merge(ovx.rename(columns={"value": "ovx"}), on="date", how="left")
        .merge(bdti.rename(columns={"value": "bdti"}), on="date", how="left")
        .merge(tanker.rename(columns={"value": "tanker_index"}), on="date", how="left")
        .merge(inventory.rename(columns={"value": "us_crude_inventory_change"}), on="date", how="left")
        .merge(gold.rename(columns={"value": "gold"}), on="date", how="left")
        .merge(vix.rename(columns={"value": "vix"}), on="date", how="left")
        .merge(hormuz.rename(columns={"value": "hormuz_transits"}), on="date", how="left")
        .sort_values("date")
    )

    df = df.ffill()

    # -------------------------
    # Z-SCORES
    # -------------------------

    df["brent_z"] = zscore(df["brent"])
    df["brent_curve_z"] = zscore(df["brent_m1_m2"])
    df["ovx_z"] = zscore(df["ovx"])
    df["bdti_z"] = zscore(df["bdti"])
    df["tanker_z"] = zscore(df["tanker_index"])
    df["inventory_draw_z"] = -zscore(df["us_crude_inventory_change"])
    df["gold_z"] = zscore(df["gold"])
    df["vix_z"] = zscore(df["vix"])
    df["transit_drop_z"] = -zscore(df["hormuz_transits"])

    # -------------------------
    # CONTRIBUTIONS
    # -------------------------

    df["contrib_brent"] = 0.10 * df["brent_z"]
    df["contrib_brent_curve"] = 0.20 * df["brent_curve_z"]
    df["contrib_ovx"] = 0.15 * df["ovx_z"]
    df["contrib_bdti"] = 0.15 * df["bdti_z"]
    df["contrib_tanker"] = 0.10 * df["tanker_z"]
    df["contrib_inventory"] = 0.08 * df["inventory_draw_z"]
    df["contrib_gold"] = 0.05 * df["gold_z"]
    df["contrib_vix"] = 0.07 * df["vix_z"]
    df["contrib_transits"] = 0.10 * df["transit_drop_z"]

    df["risk_raw"] = (
        df["contrib_brent"]
        + df["contrib_brent_curve"]
        + df["contrib_ovx"]
        + df["contrib_bdti"]
        + df["contrib_tanker"]
        + df["contrib_inventory"]
        + df["contrib_gold"]
        + df["contrib_vix"]
        + df["contrib_transits"]
    )

    df["risk_score"] = minmax_0_100(df["risk_raw"])

    df["risk_regime"] = pd.cut(
        df["risk_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["Normal", "Elevated", "Disruption Risk", "Crisis"]
    )

    save_csv(df, "dashboard_input.csv")

    latest = df.dropna(subset=["risk_score"]).iloc[-1]

    print("\nLatest snapshot")
    print("----------------")
    print(f"Date: {latest['date']}")
    print(f"Risk score: {latest['risk_score']:.1f}")
    print(f"Risk regime: {latest['risk_regime']}")

    contrib_map = {
        "Brent": latest["contrib_brent"],
        "Brent curve": latest["contrib_brent_curve"],
        "OVX": latest["contrib_ovx"],
        "BDTI": latest["contrib_bdti"],
        "Tanker": latest["contrib_tanker"],
        "Inventory": latest["contrib_inventory"],
        "Gold": latest["contrib_gold"],
        "VIX": latest["contrib_vix"],
        "Hormuz": latest["contrib_transits"],
    }

    print("\nFeature contributions")
    print("---------------------")

    for k, v in sorted(contrib_map.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:15s} {v:.3f}")

    driver = max(contrib_map, key=contrib_map.get)
    print(f"\nTop driver: {driver}")


if __name__ == "__main__":
    main()