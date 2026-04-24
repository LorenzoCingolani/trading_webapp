import streamlit as st
from p_pages import main_analysis_page, validation_page, pdm_page, forecast_page, sharpe_ratio_page

st.set_page_config(page_title="Trading App", layout="wide")
st.title("Trading Analytics")
st.write("Use the tabs below to keep each step separate and preserve completed results until you choose to rerun.")

tabs = st.tabs(["Main Analysis", "Validation", "PDM", "Forecast", "Sharpe Ratio"])

with tabs[0]:
    main_analysis_page.run()
with tabs[1]:
    validation_page.run()
with tabs[2]:
    pdm_page.run()
with tabs[3]:
    forecast_page.run()
with tabs[4]:
    sharpe_ratio_page.run()