import numpy as np
import pandas as pd
import streamlit as st

def calculate_instrument_sharpes(csvs_dictionary, framework_dict, strategy_column_map):
    sharpes = {}
    returns_dict = {}
    weights = {}
    for inst, df in csvs_dictionary.items():
        strat_col = strategy_column_map.get(inst)
        if strat_col and strat_col in df.columns:
            returns = df[strat_col].dropna()
            if not returns.empty:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                sharpes[inst] = sharpe
                returns_dict[inst] = returns
                weights[inst] = framework_dict[inst]['INSTRUMENT_WEIGHTS']
            else:
                sharpes[inst] = np.nan
        else:
            sharpes[inst] = np.nan
    return sharpes, returns_dict, weights

def calculate_portfolio_sharpe(returns_dict, weights):
    df = pd.DataFrame(returns_dict)
    df = df.dropna()
    if df.empty:
        return np.nan
    w = np.array([weights[inst] for inst in df.columns])
    w = w / w.sum()
    port_ret = df.values @ w
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    return sharpe

def run_sharpe_ratio_page(csvs_dictionary, framework_dict):
    st.title("Sharpe Ratio Analysis")

    # Let user select instruments and strategy columns
    instruments = list(csvs_dictionary.keys())
    selected_instruments = st.multiselect("Select instruments", instruments, default=instruments)
    strategy_column_map = {}
    for inst in selected_instruments:
        df = csvs_dictionary[inst]
        # Suggest columns containing 'return' or 'forecast'
        candidates = [c for c in df.columns if 'return' in c.lower() or 'forecast' in c.lower()]
        default_col = candidates[0] if candidates else df.columns[0]
        col = st.selectbox(f"Select strategy/returns column for {inst}", df.columns, index=df.columns.get_loc(default_col))
        strategy_column_map[inst] = col

    sharpes, returns_dict, weights = calculate_instrument_sharpes(
        {k: csvs_dictionary[k] for k in selected_instruments},
        framework_dict,
        strategy_column_map
    )
    st.subheader("Sharpe Ratios per Instrument")
    st.json(sharpes)

    if returns_dict:
        port_sharpe = calculate_portfolio_sharpe(returns_dict, weights)
        st.subheader("Portfolio Sharpe Ratio")
        st.write(port_sharpe)
    else:
        st.warning("No valid returns data for portfolio Sharpe calculation.")