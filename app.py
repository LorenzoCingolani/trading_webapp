import streamlit as st
from pages import main_analysis_page, validation_page, pdm_page, forecast_page

st.set_page_config(page_title="Trading App", layout="wide")

page = st.sidebar.radio("Go to", ["Main Analysis", "Validation", "PDM", "Forecast"])

if page == "Main Analysis":
    main_analysis_page.run()
elif page == "Validation":
    validation_page.run()
elif page == "PDM":
    pdm_page.run()
elif page == "Forecast":
    forecast_page.run()
