import os
import numpy as np
import pandas as pd
from typing import Dict, List
import streamlit as st

from strategies import save
from steps.multiplier_utils import diversification_multiplier
from steps.pipeline_info import show_multiplier_correlation_audit
from steps.returns import current_price_of, pct_returns

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

    all_px_closes: Dict[str, pd.Series] = {}
    all_held: Dict[str, pd.Series] = {}
    skipped: List[str] = []
    for instrument in ProductsList:
        st.write(f'Processing instrument: {instrument}')
        try:
            data = csv_dictionary[instrument]
            px = get_col_data(data, 'PX_CLOSE_1D', 'Date', "%d/%m/%Y")
            held = current_price_of(data)  # data is now indexed by Date, like px
            all_px_closes[instrument] = px
            # No near/current_price column: PX_CLOSE_1D is a real price, so it's its own divisor
            all_held[instrument] = px if held is None else held
        except Exception as ex:
            skipped.append(f"{instrument} ({ex})")

    if skipped:
        st.warning(
            "Skipping instrument(s) with bad data - fix on the Settings tab's 'Validate data' "
            "check, or deactivate them there: " + "; ".join(skipped)
        )

    ProductsList = list(all_px_closes.keys())
    if not ProductsList:
        st.error("No instruments had usable data - PDM cannot be calculated.")
        return float('nan')

    ProductsWeights: List[float] = [
        fm[instrument]['INSTRUMENT_WEIGHTS'] for instrument in ProductsList
    ]
    no_weight = [inst for inst, w in zip(ProductsList, ProductsWeights) if pd.isna(w)]
    if no_weight:
        # Don't save: a NaN PDM would overwrite a good PDM_portfolio.h5
        st.error("PDM not calculated - no instrument weight for: " + ", ".join(no_weight) +
                 ". Set weights on the Settings tab and save (then rerun Strategy Analysis).")
        return float('nan')

    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    held_df = pd.concat(all_held.values(), axis=1, keys=all_held.keys())
    # Back-adjusted price change / held-contract price (= pct_change for files without 'near')
    px_close_pct_df = pct_returns(px_close_df, held_df).dropna()

    Cmat = px_close_pct_df.corr()

    # Negative off-diagonal correlations are floored at zero for the multiplier only; Cmat (raw)
    # is what's displayed and saved as CorrMat.
    wv = np.array(ProductsWeights)
    pdm_result = diversification_multiplier(Cmat, wv, PDM_UPPER_BOUND)
    show_multiplier_correlation_audit(pdm_result)

    PDM_original = pdm_result.uncapped_multiplier
    PDM_capped = pdm_result.multiplier

    st.write(f"Original PDM: {PDM_original:.4f}")
    st.write(f"Capped PDM (max {PDM_UPPER_BOUND}): {PDM_capped:.4f}")

    Out = save.Output('pdm')
    Out.products_list = ProductsList
    Out.products_weights = ProductsWeights
    Out.CorrMat = Cmat.values
    Out.CorrMatMultiplier = pdm_result.multiplier_corr.values
    Out.portfolio_diver_mult = PDM_capped
    Out.portfolio_diver_mult_uncapped = PDM_original

    savecode = 'PDM_portfolio.h5'
    path = os.path.join('DATA', 'combinedForecast')
    os.makedirs(path, exist_ok=True)
    save.h5file(path, savecode, *(Out,))
    st.success(f'Successfully computed and saved PDM (capped): {PDM_capped:.4f} to {os.path.join(path, savecode)}')
    return PDM_capped