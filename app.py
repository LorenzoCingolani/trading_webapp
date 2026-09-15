import streamlit as st
from p_pages import settings_page, main_analysis_page, validation_page, pdm_page, forecast_page, sharpe_ratio_page
from steps.checkpoint import create_checkpoint

st.set_page_config(page_title="Trading App", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #0F172A;
}

h1 { border-bottom: 1px solid #E2E8F0; padding-bottom: 0.5rem; }

p, li, label, .stMarkdown { color: #334155; }

hr { margin: 1.6rem 0; border-color: #E2E8F0; }

/* Buttons */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    border: 1px solid #E2E8F0;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.10);
}
div.stButton > button[kind="primary"] {
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45);
}

/* Top page-nav pills get a bit more room and rounding */
div[data-testid="column"] div.stButton > button {
    border-radius: 10px;
    padding: 0.6rem 0.5rem;
}

/* Inputs / selects */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    border-radius: 8px !important;
}

/* Expanders read as clean cards */
div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    background-color: #F8FAFC;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    color: #2563EB;
    font-weight: 700;
}

/* Alert boxes */
div[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Dataframes/tables */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

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
