import numpy as np
import pandas as pd
import streamlit as st

def calculate_instrument_sharpes(csvs_dictionary, framework_dict, strategy_column_map):
    sharpes = {}         # Store Sharpe ratios per instrument
    returns_dict = {}    # Store returns series per instrument
    weights = {}         # Store weights per instrument
    for inst, df in csvs_dictionary.items():
        strat_col = strategy_column_map.get(inst)  # Get selected strategy column for this instrument
        if strat_col and strat_col in df.columns:
            returns = df[strat_col].dropna()       # Drop missing values from returns
            if not returns.empty:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe ratio
                sharpes[inst] = sharpe
                returns_dict[inst] = returns
                weights[inst] = framework_dict[inst]['INSTRUMENT_WEIGHTS']
            else:
                sharpes[inst] = np.nan             # If no data, set as NaN
        else:
            sharpes[inst] = np.nan                 # If column not found, set as NaN
    return sharpes, returns_dict, weights

def calculate_portfolio_sharpe(returns_dict, weights):
    df = pd.DataFrame(returns_dict)                # Combine all returns into a DataFrame (aligned by date)
    df = df.dropna()                               # Drop rows with any missing values
    if df.empty:
        return np.nan
    w = np.array([weights[inst] for inst in df.columns])  # Get weights in correct order
    w = w / w.sum()                                # Normalize weights to sum to 1
    port_ret = df.values @ w                       # Calculate weighted portfolio returns
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)  # Portfolio Sharpe ratio
    return sharpe, port_ret, df.index              # Also return portfolio returns and dates for plotting

def run_sharpe_ratio_page(csvs_dictionary, framework_dict):
    st.title("Sharpe Ratio Analysis")

    # Let user select instruments and strategy columns
    instruments = list(csvs_dictionary.keys())
    selected_instruments = st.multiselect(
        "Select instruments", instruments, default=instruments
    )
    strategy_column_map = {}
    for inst in selected_instruments:
        df = csvs_dictionary[inst]
        # Suggest columns containing 'return' or 'forecast'
        candidates = [c for c in df.columns if 'return' in c.lower() or 'forecast' in c.lower()]
        default_col = candidates[0] if candidates else df.columns[0]
        col = st.selectbox(
            f"Select strategy/returns column for {inst}",
            df.columns,
            index=df.columns.get_loc(default_col)
        )
        strategy_column_map[inst] = col

    # Calculate Sharpe ratios and collect returns
    sharpes, returns_dict, weights = calculate_instrument_sharpes(
        {k: csvs_dictionary[k] for k in selected_instruments},
        framework_dict,
        strategy_column_map
    )

    # Show Sharpe ratios as a table and as JSON
    st.subheader("Sharpe Ratios per Instrument")
    st.dataframe(pd.DataFrame.from_dict(sharpes, orient='index', columns=['Sharpe Ratio']))
    st.json(sharpes)

    # Show returns time series for each instrument
    if returns_dict:
        st.subheader("Returns Time Series (first 10 rows)")
        returns_df = pd.DataFrame(returns_dict)
        st.dataframe(returns_df.head(10))

        # Plot returns for each instrument
        st.subheader("Returns Plot per Instrument")
        for inst, series in returns_dict.items():
            st.line_chart(series.rename(inst))

        # Calculate and show portfolio Sharpe ratio and plot
        port_sharpe, port_ret, port_dates = calculate_portfolio_sharpe(returns_dict, weights)
        st.subheader("Portfolio Sharpe Ratio")
        st.write(port_sharpe)

        # Plot portfolio cumulative returns
        st.subheader("Portfolio Cumulative Returns")
        port_cum = np.cumsum(port_ret)
        port_cum_df = pd.DataFrame({'Cumulative Return': port_cum}, index=port_dates)
        st.line_chart(port_cum_df)

        # Show correlation matrix between instruments
        st.subheader("Correlation Matrix of Returns")
        st.dataframe(returns_df.corr())

    else:
        st.warning("No valid returns data for portfolio Sharpe calculation.")