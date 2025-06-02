import os
import numpy as np
import pandas as pd
from typing import Dict, List
import streamlit as st

from strategies import save

PDM_UPPER_BOUND = 2

def get_col_data(
    data: pd.DataFrame,
    col_name: str,
    date_col: str = 'Date',
    date_format: str = '%Y-%m-%d'
) -> pd.Series:
    data.dropna(subset=[date_col], inplace=True)
    try:
        data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    except ValueError:
        data[date_col] = pd.to_datetime(data[date_col], format='mixed', dayfirst=True)
    data.set_index(date_col, inplace=True)
    return data[col_name].copy()

def pdm_main(fm: Dict, csv_dictionary: Dict[str, pd.DataFrame]) -> float:
    """
    Compute the Portfolio Diversification Multiplier (PDM) from product data and display results in Streamlit.

    Args:
        fm (Dict): Framework dictionary containing instrument weights.
        csv_dictionary (Dict[str, pd.DataFrame]): Dictionary mapping instrument names to DataFrames.

    Returns:
        float: The capped PDM value.
    """
    st.subheader("Portfolio Diversification Multiplier (PDM) Calculation")
    ProductsList: List[str] = list(csv_dictionary.keys())
    st.write('Products in portfolio:', ProductsList)

    ProductsWeights: List[float] = [
        fm[instrument]['INSTRUMENT_WEIGHTS'] for instrument in ProductsList
    ]

    all_px_closes: Dict[str, pd.Series] = {}
    for instrument in ProductsList:
        st.write(f'Processing instrument: {instrument}')
        all_px_closes[instrument] = get_col_data(
            csv_dictionary[instrument], 'PX_CLOSE_1D', 'Date', "%d/%m/%Y"
        )

    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    px_close_pct_df = px_close_df.pct_change().dropna()

    Cmat = px_close_pct_df.corr()
    st.write("Correlation matrix:")
    st.dataframe(Cmat)

    wv = np.array(ProductsWeights)
    PDM_original = 1.0 / np.sqrt(np.dot(wv.T, np.dot(Cmat.values, wv)))
    PDM_capped = min(PDM_original, PDM_UPPER_BOUND)

    st.write(f"Original PDM: {PDM_original:.4f}")
    st.write(f"Capped PDM (max {PDM_UPPER_BOUND}): {PDM_capped:.4f}")

    Out = save.Output('pdm')
    Out.products_list = ProductsList
    Out.products_weights = ProductsWeights
    Out.CorrMat = Cmat.values
    Out.portfolio_diver_mult = PDM_capped
    Out.portfolio_diver_mult_uncapped = PDM_original

    savecode = 'PDM_portfolio.h5'
    path = os.path.join('DATA', 'combinedForecast')
    os.makedirs(path, exist_ok=True)
    save.h5file(path, savecode, *(Out,))
    st.success(f'Successfully computed and saved PDM (capped): {PDM_capped:.4f} to {os.path.join(path, savecode)}')
    return PDM_capped