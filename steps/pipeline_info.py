import os
import pandas as pd
import streamlit as st

INPUT_MAIN_CSV = os.path.join('DATA', 'input_main', 'input_main.csv')
INPUT_INSTRUMENTS_DIR = os.path.join('DATA', 'input_instruments')


def active_instruments_df() -> pd.DataFrame:
    if os.path.isdir(INPUT_INSTRUMENTS_DIR):
        instruments = sorted(f[:-4] for f in os.listdir(INPUT_INSTRUMENTS_DIR) if f.endswith('.csv'))
    else:
        instruments = []

    weights = {}
    if os.path.exists(INPUT_MAIN_CSV):
        wdf = pd.read_csv(INPUT_MAIN_CSV)
        if 'INSTRUMENT' in wdf.columns and 'INSTRUMENT_WEIGHTS' in wdf.columns:
            weights = dict(zip(wdf['INSTRUMENT'], wdf['INSTRUMENT_WEIGHTS']))

    return pd.DataFrame([{'Instrument': inst, 'Weight': weights.get(inst)} for inst in instruments])


def show_active_instruments() -> None:
    df = active_instruments_df()
    with st.expander(f"Active instruments ({len(df)})", expanded=False):
        if df.empty:
            st.warning("No active instruments selected. Set them on the Settings tab.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Configured on the Settings tab (Instrument Pool + Instrument Weights).")


def show_source_paths(entries) -> None:
    """entries: list of 'relative/path.py :: function_name()' strings, shown up front (not collapsed)."""
    lines = "\n".join(f"- `{entry}`" for entry in entries)
    st.caption("Running code from:")
    st.markdown(lines)


def _strategy_explanations() -> dict:
    explanations = {}

    try:
        from strategies_mine.ewma_no_tick import FORECAST_SCALERS, CAP as EWMA_CAP
        pairs = ", ".join(f"{f}d/{s}d" for f, s in FORECAST_SCALERS.keys())
        explanations['EWMA'] = (
            f"Exponentially-weighted moving-average crossover: compares a fast EWMA of price to a slow "
            f"EWMA (fast/slow day pairs: {pairs}) to build a forecast, scaled per pair and capped at "
            f"±{EWMA_CAP:.0f}."
        )
    except Exception:
        pass

    try:
        from strategies_mine.strategies.carry import CARRY_SPANS, CARRY_SCALER, FORECAST_CAP as CARRY_CAP, TARGET_VOL, AUM as CARRY_AUM
        explanations['CARRY'] = (
            f"Carry: (far price - near price) normalized by volatility, smoothed over spans of "
            f"{CARRY_SPANS} days, scaled by {CARRY_SCALER:.0f} and capped at ±{CARRY_CAP:.0f}. "
            f"Position sizing targets {TARGET_VOL:.0%} annualized volatility on a ${CARRY_AUM:,.0f} notional AUM."
        )
    except Exception:
        pass

    try:
        from steps.p1_analysis import EWMA_NORM_RULES, FORECAST_CAP as NORM_CAP
        spans = ", ".join(f"{fast}d/{fast * 4}d" for fast in EWMA_NORM_RULES)
        explanations['EWMA_NORM'] = (
            f"EWMA Norm: like EWMA but computed on a volatility-normalized price series, across "
            f"fast/slow spans {spans}, capped at ±{NORM_CAP:.0f}. Each span only counts if its "
            f"implied trading cost passes a cost filter; the spans that pass are averaged and rescaled."
        )
    except Exception:
        pass

    explanations['BREAKOUT'] = (
        "Breakout: selectable here, but this page does not currently compute or save Breakout output "
        "- selecting it has no effect yet."
    )

    return explanations


STRATEGY_SOURCES = {
    'EWMA': 'strategies_mine/ewma_no_tick.py :: compute_all_ewma()',
    'CARRY': 'strategies_mine/strategies/carry.py :: calc()',
    'EWMA_NORM': 'steps/p1_analysis.py :: _compute_ewma_norm()',
    'BREAKOUT': '(not wired up - selecting it has no effect)',
}


def strategy_source_paths(selected_strategies) -> list:
    selected = {str(s).upper() for s in selected_strategies}
    return [f"{key}: {STRATEGY_SOURCES[key]}" for key in ['EWMA', 'CARRY', 'EWMA_NORM', 'BREAKOUT'] if key in selected]


def show_strategy_explanations(selected_strategies) -> None:
    explanations = _strategy_explanations()
    selected = {str(s).upper() for s in selected_strategies}
    with st.expander("What the selected strategies compute", expanded=False):
        for key in ['EWMA', 'CARRY', 'EWMA_NORM', 'BREAKOUT']:
            if key in selected and key in explanations:
                st.markdown(f"**{key}** - {explanations[key]}")


def show_step_explanation(text: str) -> None:
    with st.expander("What this step computes", expanded=False):
        st.markdown(text)
