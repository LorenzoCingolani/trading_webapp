
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="Rates Dashboard (IBKR + FRED)", layout="wide")

st.title("Fixed Income Dashboard — UST/CA/DE/UK/JP | Curves | US 5y5y (inflation)")
st.caption(
    "Live govvie benchmarks via IBKR (fill conIds in instruments CSV). "
    "US 5y5y Inflation Expectation via FRED (T5YIFR)."
)

default_csv = "instruments_example.csv"
instruments_file = st.sidebar.text_input("Instruments CSV path", value=default_csv)

use_ib = st.sidebar.checkbox("Connect to IBKR (TWS/Gateway)", value=False)
ib_host = st.sidebar.text_input("IB Host", value="127.0.0.1")
ib_port = st.sidebar.number_input("IB Port", value=7497, step=1)
ib_client_id = st.sidebar.number_input("IB Client ID", value=37, step=1)

ib = None
if use_ib:
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(ib_host, int(ib_port), clientId=int(ib_client_id), readonly=True, timeout=5)
        st.sidebar.success("Connected to IBKR")
    except Exception as e:
        st.sidebar.error(f"IBKR connect failed: {e}")
        ib = None

@st.cache_data
def load_instruments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"country","tenor","label","secType","currency","ib_conid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in instruments CSV: {missing}")
    return df

try:
    instruments = load_instruments(instruments_file)
except Exception as e:
    st.error(f"Failed to load instruments CSV: {e}")
    st.stop()

def fetch_ib_bond_yields(df: pd.DataFrame, ib):
    out_rows = []
    if ib is None:
        return pd.DataFrame(columns=["country","tenor","label","yield_pct"]).astype({"yield_pct":"float"})
    from ib_insync import Contract
    for _, r in df.iterrows():
        conid_str = str(r.get("ib_conid") or "").strip()
        if not conid_str:
            out_rows.append({**r.to_dict(), "yield_pct": np.nan})
            continue
        try:
            c = Contract(conId=int(conid_str))
            ticker = ib.reqMktData(c, "", False, False)
            ib.sleep(0.4)
            y = None
            # IB fields that may contain yield
            if getattr(ticker, "yieldBid", None) is not None and getattr(ticker, "yieldAsk", None) is not None:
                y = (ticker.yieldBid + ticker.yieldAsk) / 2.0
            elif getattr(ticker, "yield_", None) is not None:
                y = ticker.yield_
            elif getattr(ticker, "last", None) is not None and ticker.last:
                y = ticker.last
            elif getattr(ticker, "close", None) is not None and ticker.close:
                y = ticker.close
            out_rows.append({**r.to_dict(), "yield_pct": float(y) if y is not None else np.nan})
        except Exception:
            out_rows.append({**r.to_dict(), "yield_pct": np.nan})
    return pd.DataFrame(out_rows)

gov_df = fetch_ib_bond_yields(instruments, ib)

def styled(df):
    return (df[["country","tenor","label","yield_pct"]]
            .sort_values(["country","tenor"]))

st.subheader("Benchmark Yields (live via IBKR where conIds are provided)")
st.dataframe(styled(gov_df), use_container_width=True)

def compute_curves(df: pd.DataFrame, country: str):
    sub = df[df["country"] == country].copy()
    m = {"2Y":2, "5Y":5, "10Y":10, "30Y":30}
    sub["mat"] = sub["tenor"].map(m)
    sub = sub.dropna(subset=["mat","yield_pct"])
    if sub.empty:
        return None
    piv = sub.pivot_table(index="country", columns="tenor", values="yield_pct", aggfunc="mean")
    if not set(["2Y","10Y","30Y"]).issubset(piv.columns):
        return None
    s2s10 = piv["10Y"].iloc[0] - piv["2Y"].iloc[0]
    s2s30 = piv["30Y"].iloc[0] - piv["2Y"].iloc[0]
    return s2s10, s2s30, piv

countries = ["USA","Canada","Germany","UK","Japan"]
curve_rows = []
for c in countries:
    res = compute_curves(gov_df, c)
    if res:
        s2s10, s2s30, piv = res
        curve_rows.append({"country": c, "2s10s_bp": s2s10*100, "2s30s_bp": s2s30*100})
curves_df = pd.DataFrame(curve_rows)

col1, col2 = st.columns(2)
with col1:
    st.subheader("2s10s (bp)")
    st.dataframe(curves_df[["country","2s10s_bp"]], use_container_width=True)
with col2:
    st.subheader("2s30s (bp)")
    st.dataframe(curves_df[["country","2s30s_bp"]], use_container_width=True)

st.subheader("US 5y5y Inflation Expectation — FRED:T5YIFR")
start_date = st.sidebar.date_input("5y5y Start Date", value=(datetime.utcnow() - timedelta(days=365*5)).date())

def get_fred_5y5y(start):
    try:
        import pandas_datareader.data as web
        fred = web.DataReader("T5YIFR","fred", start, datetime.utcnow())
        fred = fred.rename(columns={"T5YIFR":"5y5y_inflation"}).dropna()
        fred.index = pd.to_datetime(fred.index)
        return fred
    except Exception as e:
        st.warning(f"FRED fetch failed: {e}")
        return pd.DataFrame()

fred_df = get_fred_5y5y(start_date)
if not fred_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fred_df.index, y=fred_df["5y5y_inflation"], mode="lines", name="US 5y5y"))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Date", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No FRED data to display yet.")

st.markdown("---")
st.caption("Tip: In IB TWS, use Contract Search to find **benchmark** government bonds (on-the-run). "
           "Copy their conIds into the CSV, then toggle 'Connect to IBKR'.")
