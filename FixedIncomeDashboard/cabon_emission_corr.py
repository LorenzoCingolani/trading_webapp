# cabon_emission_corr.py
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

TICKERS = [
    "TSLA",        # Tesla
    "IBE.MC",      # Iberdrola (Spain)
    "BEP",         # Brookfield Renewable Partners
    "ORSTED.CO",   # Orsted (Denmark)
    "VWS.CO",      # Vestas Wind Systems (Denmark)  # <-- FIXED
    "ENEL.MI",     # Enel (Italy)
    "ENR.DE",      # Siemens Energy (Germany)
    "NEE",         # NextEra Energy (US)
    "ORA",         # Ormat Technologies (US)
    "BLDP",        # Ballard Power Systems (Canada)
    "CARB.L"         # Carbon price proxy ETF (global carbon futures)
]

CARBON_PROXY = "CARB.L"
CARBON_FALLBACK = "KRBN"   # Carbon price proxy ETF (global carbon futures)

START = "2022-01-01"
END   = "2025-11-01"
OUTFILE = "carbon_correlation_analysis.xlsx"

def download_close_prices(tickers):
    # auto_adjust=True -> adjusted OHLCV with NO 'Adj Close'. Use 'Close'.
    df = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Handle single vs multi-index returns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Expect ('<TICKER>', 'Close') or ('Close', '<TICKER>') depending on version
        # Normalize to a flat DataFrame of Close prices with ticker columns.
        close_frames = []
        for t in tickers:
            try:
                # Common layout: df[t]['Close']
                s = df[t]["Close"].rename(t)
                close_frames.append(s)
            except Exception:
                # Alternate layout: df['Close'][t]
                try:
                    s = df["Close"][t].rename(t)
                    close_frames.append(s)
                except Exception:
                    # Skip if not present
                    pass
        if close_frames:
            px = pd.concat(close_frames, axis=1)
        else:
            px = pd.DataFrame()
    else:
        # Single-index columns: could be just 'Close' for single ticker
        # If multiple tickers returned flat columns, keep as-is
        if "Close" in df.columns:
            px = df["Close"].to_frame()
            # When single ticker requested, name the column as that ticker
            if len(tickers) == 1:
                px.columns = [tickers[0]]
        else:
            # If already flat prices per ticker (rare), keep df
            px = df.copy()

    # Drop columns that are all-NaN
    px = px.dropna(axis=1, how="all")
    return px

# 1) First download
prices = download_close_prices(TICKERS)
got = list(prices.columns)
missing = [t for t in TICKERS if t not in got]

print(f"✅ Downloaded tickers: {got}")
if missing:
    print(f"⚠️ Missing (no data): {missing}")

# 2) Ensure we have a carbon proxy; if KRBN missing, try CARB.L
if CARBON_PROXY not in prices.columns:
    print(f"⚠️ '{CARBON_PROXY}' missing. Trying fallback '{CARBON_FALLBACK}'...")
    fb = download_close_prices([CARBON_FALLBACK])
    if not fb.empty and CARBON_FALLBACK in fb.columns:
        prices = prices.join(fb, how="outer")
        CARBON_PROXY = CARBON_FALLBACK
        print(f"✅ Using '{CARBON_FALLBACK}' as carbon proxy.")
    else:
        print("❌ Fallback carbon proxy download failed.")

# 3) Compute returns & correlations
returns = prices.sort_index().pct_change().dropna(how="all")
corr_matrix = returns.corr()

# 4) Extract correlation vs carbon proxy (if present)
if CARBON_PROXY in corr_matrix.columns:
    cp_corr = corr_matrix[CARBON_PROXY].sort_values(ascending=False)
else:
    cp_corr = pd.Series(dtype=float)
    print(f"⚠️ Carbon proxy '{CARBON_PROXY}' not present in correlation matrix.")

# 5) Save to Excel
with pd.ExcelWriter(OUTFILE) as writer:
    corr_matrix.to_excel(writer, sheet_name="Correlation_Matrix")
    cp_corr.to_excel(writer, sheet_name="CarbonProxy_Correlations")

print(f"✅ Saved results to '{OUTFILE}'")

# 6) Plot heatmap if non-empty
if not corr_matrix.empty:
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, interpolation='none')  # no explicit colors
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title("Correlation Matrix (Stocks vs Carbon Proxy)")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Correlation matrix is empty — nothing to plot.")
