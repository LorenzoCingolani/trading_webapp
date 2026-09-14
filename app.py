import streamlit as st
from p_pages import settings_page, main_analysis_page, validation_page, pdm_page, forecast_page, sharpe_ratio_page
from steps.checkpoint import create_checkpoint

st.set_page_config(page_title="Trading App", layout="wide")

if 'startup_checkpoint_done' not in st.session_state:
    create_checkpoint(reason='startup')
    st.session_state.startup_checkpoint_done = True

PAGES = [
    ("Settings", settings_page),
    ("Main Analysis", main_analysis_page),
    ("Validation", validation_page),
    ("PDM", pdm_page),
    ("Forecast", forecast_page),
    ("Sharpe Ratio", sharpe_ratio_page),
]
PAGE_NAMES = [name for name, _ in PAGES]

if 'current_page' not in st.session_state:
    st.session_state.current_page = PAGE_NAMES[0]

st.title("Trading Analytics")
st.write("Use the buttons below to keep each step separate and preserve completed results until you choose to rerun.")


def _go_to(name: str) -> None:
    st.session_state.current_page = name
    st.rerun()


nav_cols = st.columns(len(PAGE_NAMES))
for i, name in enumerate(PAGE_NAMES):
    with nav_cols[i]:
        is_current = name == st.session_state.current_page
        if st.button(name, key=f"nav_top_{name}", type="primary" if is_current else "secondary", use_container_width=True):
            _go_to(name)

st.divider()

current_idx = PAGE_NAMES.index(st.session_state.current_page)
PAGES[current_idx][1].run()

st.divider()
col_back, col_spacer, col_next = st.columns([1, 6, 1])
with col_back:
    if current_idx > 0:
        if st.button("< Back", key="nav_back", use_container_width=True):
            _go_to(PAGE_NAMES[current_idx - 1])
with col_next:
    if current_idx < len(PAGE_NAMES) - 1:
        if st.button("Next >", key="nav_next", use_container_width=True):
            _go_to(PAGE_NAMES[current_idx + 1])
