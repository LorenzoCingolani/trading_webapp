
# ## Portfolio diversification multiplier 
# 
#  This is computed starting from the correlation between products, so it 
# requires to have `run scan_products.py` first. It generates a file containing the PDM, 
# which is required when running framework.


import os

import numpy as np
import pandas as pd

from .strategies import save
from django.conf import settings

PDM_UPPER_BOUND = 2


def get_col_data(data, col_name,date_col='Date',date_format='%Y-%m-%d'):
    data.dropna(subset=[date_col],inplace=True)
    try:
        data[date_col] = pd.to_datetime(data[date_col], format=date_format)
    except ValueError:
        data[date_col] = pd.to_datetime(data[date_col], format='mixed', dayfirst=True)
    data.set_index(date_col, inplace=True)
    return data[col_name].copy()


def pdm_main(fm, csv_dictionary):
    
    ProductsList= list(csv_dictionary.keys())
    print('ProductsList:', ProductsList)
    ProductsWeights=[]
    for ins in ProductsList:
        # get the weights from the framework file
        weight = fm[ins]['INSTRUMENT_WEIGHTS']
        ProductsWeights.append(weight)
    all_px_closes = {}
    for instrument in ProductsList:
        print('Instrument:', instrument)
        print('csv_dictionary[instrument]:', csv_dictionary[instrument])
        all_px_closes[instrument] = get_col_data(csv_dictionary[instrument], 'PX_CLOSE_1D', 'Date', "%d/%m/%Y")
    px_close_df = pd.concat(all_px_closes.values(), axis=1, keys=all_px_closes.keys())

    px_close_pct_df = px_close_df.pct_change()
    Cmat = px_close_pct_df.corr()
    ### Portfolio Diversification Multiplier
    wv=np.array(ProductsWeights)
    PDM=1./np.sqrt(np.dot(wv.T,np.dot(Cmat,wv)))
    PDM = min(PDM, PDM_UPPER_BOUND)
    # general output
    Out=save.Output('pdm')
    Out.products_list=ProductsList
    Out.products_weights=ProductsWeights
    Out.CorrMat=Cmat.values
    Out.portfolio_diver_mult=PDM
    savecode='PDM_portfolio.h5'
    path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')
    save.h5file(os.path.join(path),savecode,*(Out,))
    print('successfully computed PDM:', PDM)


if __name__ == '__main__':
    # Read products and weights from framewrok input file
    framework_input_file='./Files/input_framework_port - Original - Test.csv'
 
    MainFolderPath='./Files'
    csv_dictionary=MainFolderPath+'/all_in_1year/portfolio1'
    fm=pd.read_csv(framework_input_file)
    pdm_main(fm, csv_dictionary)



