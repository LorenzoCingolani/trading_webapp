"""
Portfolio Diversification Multiplier (PDM) Calculation Module

This module calculates the Portfolio Diversification Multiplier (PDM) based on
the correlation between product returns and their weights in the portfolio.

Dependencies:
- Requires product price data and weights from scan_products.py.
- Outputs a file `PDM_portfolio.h5` containing PDM and related metadata.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List

from .strategies import save
from django.conf import settings

PDM_UPPER_BOUND = 2


def get_col_data(
    data: pd.DataFrame,
    col_name: str,
    date_col: str = 'Date',
    date_format: str = '%Y-%m-%d'
) -> pd.Series:
    """
    Extract a specific column of data after parsing the date column and setting it as index.

    Args:
        data (pd.DataFrame): Raw input data containing the required columns.
        col_name (str): Name of the column to extract.
        date_col (str, optional): Name of the column containing date values. Defaults to 'Date'.
        date_format (str, optional): Expected date format. Defaults to '%Y-%m-%d'.

    Returns:
        pd.Series: Time-indexed series for the specified column.
    """
    data.dropna(subset=[date_col], inplace=True)
    try:
        data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    except ValueError:
        data[date_col] = pd.to_datetime(data[date_col], format='mixed', dayfirst=True)
    data.set_index(date_col, inplace=True)
    return data[col_name].copy()


def pdm_main(fm: Dict, csv_dictionary: Dict[str, pd.DataFrame]) -> None:
    """
    Compute the Portfolio Diversification Multiplier (PDM) from product data.

    Args:
        fm (Dict): Framework dictionary containing instrument weights.
        csv_dictionary (Dict[str, pd.DataFrame]): Dictionary mapping instrument names to DataFrames.

    Outputs:
        HDF5 file containing the PDM and correlation matrix saved to disk.
    """
    ProductsList: List[str] = list(csv_dictionary.keys())
    print('ProductsList:', ProductsList)

    ProductsWeights: List[float] = [
        fm[instrument]['INSTRUMENT_WEIGHTS'] for instrument in ProductsList
    ]

    all_px_closes: Dict[str, pd.Series] = {}
    for instrument in ProductsList:
        print('Instrument:', instrument)
        print('csv_dictionary[instrument]:', csv_dictionary[instrument])
        all_px_closes[instrument] = get_col_data(
            csv_dictionary[instrument], 'PX_CLOSE_1D', 'Date', "%d/%m/%Y"
        )

    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())
    px_close_pct_df = px_close_df.pct_change().dropna()

    Cmat = px_close_pct_df.corr()

    wv = np.array(ProductsWeights)
    PDM = 1.0 / np.sqrt(np.dot(wv.T, np.dot(Cmat.values, wv)))
    PDM = min(PDM, PDM_UPPER_BOUND)

    Out = save.Output('pdm')
    Out.products_list = ProductsList
    Out.products_weights = ProductsWeights
    Out.CorrMat = Cmat.values
    Out.portfolio_diver_mult = PDM

    savecode = 'PDM_portfolio.h5'
    path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')
    save.h5file(path, savecode, *(Out,))
    print('Successfully computed PDM:', PDM)
    return PDM
