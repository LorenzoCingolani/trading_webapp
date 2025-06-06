import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from . import save

from collections import namedtuple

def calc_turnover(ewma_slow, data, aum=10000000):
    data['1%_move'] = data['PX_CLOSE_1D']*0.01
    data['block_value'] = data['1%_move']*data['point_value']
    data['price_volatility'] = round(data['st_dev']/data['PX_CLOSE_1D']*100,2)
    data['ICV'] = data['price_volatility'] * data['block_value']
    data['IVV'] = data['ICV'] * data['exchange_rate']
    data['Daily_Cash_Vol_Tgt'] = aum*.2/16
    data['Volatility_Scalar'] = data['Daily_Cash_Vol_Tgt']/data['IVV']
    data['Subsystem_Pos'] = data['Volatility_Scalar'] * data['capped_forecast']/10.
    data['Target_Pos'] =  data['Subsystem_Pos'].round(decimals=0)
    data['Target_Pos'].fillna(0, inplace=True)
    data['Current_pos'] = data['Target_Pos'].shift()
    data['trades_needed']=data['Target_Pos']-data['Current_pos']
    avg_abs_valtgtpos= abs(data['Current_pos']).mean()
    sum_abs_trades_needed= abs(data['trades_needed']).sum()
    years=data.shape[0] / 256
    trades_needed_yearly=sum_abs_trades_needed/years
    turnover=trades_needed_yearly/(2*avg_abs_valtgtpos)
    return turnover

def calc(Inst_name, data, MAParam, standard_cost, exchange_rate=1.0, point_value=50):
    st.write('Your MAParam is', MAParam)

    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value

    all_ma_fast_df = pd.DataFrame()
    all_abs_capped_forecast_df = pd.DataFrame(
        columns=('EWMA', 'Avg_Capped_Forecast'))

    EWMAList = []
    CumList = []   # list to store cumulative series (plot only, not saving)
    DaysList = []  # list to store days (for plot only)

    strategies_passed_names = []

    for ewma_fast in MAParam:
        EWMAhere = EWMAout(ewma_fast)
        EWMAhere.TH = save.TimeHistory()
        EWMAhere.TH.px_close = np.array(data['PX_CLOSE_1D'])
        EWMAhere.TH.st_dev = np.array(data['st_dev'])
        EWMAhere.TH.start_date = data.values[0, 0]

        st.write(f"Calculating ewma_fast {ewma_fast}")
        ewma_slow = ewma_fast*4
        stdev_lookback = 36
        data['stdev_lookback'] = stdev_lookback
        data['stdev_decay'] = 2/(stdev_lookback+1)
        data['decay_fast'] = 2/(ewma_fast+1)
        data['decay_slow'] = 2/(ewma_slow+1)

        if ewma_fast == 2:
            forecast_scalar = 10.6
        elif ewma_fast == 4:
            forecast_scalar = 7.5
        elif ewma_fast == 8:
            forecast_scalar = 5.3
        elif ewma_fast == 16:
            forecast_scalar = 3.75
        elif ewma_fast == 32:
            forecast_scalar = 2.65
        elif ewma_fast == 64:
            forecast_scalar = 1.87
        else:
            raise Exception("Invalid value for ewma_fast")
        data['Daily_Return'] = data['PX_CLOSE_1D'].ffill().pct_change()
        data['returns'] = data['PX_CLOSE_1D'].diff()
        data.loc[0, 'returns'] = 0.
        data['sqreturns'] = data['returns']*data['returns']

        data.loc[0, 'emwa_fast_fin'] = data.loc[0, 'PX_CLOSE_1D']
        for i in range(1, len(data)):
            data.loc[i, 'emwa_fast_fin'] = (data.loc[i, 'decay_fast'] *
                                            data.loc[i, 'PX_CLOSE_1D']+(data.loc[i-1, 'emwa_fast_fin']
                                                                        * (1-data.loc[i, 'decay_fast'])))

        data.loc[0, 'emwa_slow_fin'] = data.loc[0, 'PX_CLOSE_1D']
        for i in range(1, len(data)):
            data.loc[i, 'emwa_slow_fin'] = (data.loc[i, 'decay_slow'] *
                                            data.loc[i, 'PX_CLOSE_1D']+(data.loc[i-1, 'emwa_slow_fin']
                                                                        * (1-data.loc[i, 'decay_slow'])))

        data['raw_cross'] = data['emwa_fast_fin']-data['emwa_slow_fin']
        data['variance'] = 0.
        data.loc[1, 'variance'] = data.loc[1, 'sqreturns']
        for i in range(2, len(data)):
            data.loc[i, 'variance'] = data.loc[i, 'stdev_decay']\
                * data.loc[i, 'sqreturns']+(1-data.loc[i, 'stdev_decay'])\
                * data.loc[i-1, 'variance']

        data['std_dev'] = data['variance']**.5
        data['vol_adj_crossover'] = data['raw_cross']/data['std_dev']
        data['forecast_scalar'] = forecast_scalar
        data['forecast'] = data['vol_adj_crossover']*forecast_scalar
        data['capped_forecast'] = data['forecast'].clip(-20, +20)
        avg_abs_val_capped_forecast = abs(data['capped_forecast']).mean()

        turnover = calc_turnover(ewma_slow, data)
        data['abs_forecast'] = abs(data['forecast'])
        data['forecast*returns'] = data['capped_forecast']*data['Daily_Return'].shift(-1)
        data['cum_series'] = data['forecast*returns'].cumsum()

        all_ma_fast_df['cum_series_' + str(ewma_fast)] = data['cum_series']
        all_abs_capped_forecast_df.loc[len(all_abs_capped_forecast_df)] =\
            [ewma_fast, abs(data['capped_forecast']).mean()]

        ewma_gross_ret_stedv = np.std(data['forecast*returns'])
        ewma_gross_ret_mean = np.mean(data['forecast*returns'])
        ewma_gross_ret_sr = ewma_gross_ret_mean*np.sqrt(256)/ewma_gross_ret_stedv
        
        if ewma_gross_ret_sr > 0.:
            ewma_net_ret_sr = ewma_gross_ret_sr-(ewma_gross_ret_sr*standard_cost)
        else:
            ewma_net_ret_sr = ewma_gross_ret_sr + (ewma_gross_ret_sr*standard_cost)   

        years = data.shape[0] / 256
        max_payable = 0.13/turnover

        if max_payable < standard_cost:
            EWMAhere.speed = 'Expensive_discard'
        else:
            EWMAhere.speed = 'Good_trade'
            # Save CSV to current working directory
            out_csv = os.path.join(
                os.getcwd(), 'DATA', 'output_instruments', f'{Inst_name}_ewma_{EWMAhere.name}.csv')
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            data.to_csv(out_csv)
            strategies_passed_names.append(EWMAhere.name)
            st.success(f'Saved csv to path {out_csv}')

        EWMAhere.standard_cost = standard_cost
        EWMAhere.avg_abs_val_capped_forecast = avg_abs_val_capped_forecast
        EWMAhere.ewma_gross_ret_sr = ewma_gross_ret_sr
        EWMAhere.ewma_net_ret_sr = ewma_net_ret_sr
        EWMAhere.turnover = turnover
        EWMAhere.forecast_scalar = forecast_scalar
        EWMAhere.max_payable = max_payable
        EWMAhere.years = years
        EWMAhere.ewma_gross_ret_sr  = ewma_gross_ret_sr 
        EWMAhere.cum_series = np.array(data.cum_series.copy()[:])  # cum series
        EWMAList.append(EWMAhere)

        # store variables for plotting
        CumList.append(data['cum_series'])
        DaysList.append(np.arange(1, len(data)+1))

    # Plot all cumulative series and show in Streamlit
    plot_cum_series_streamlit(MAParam, DaysList, CumList)
    EWMAList.append(strategies_passed_names)

    return EWMAList

def plot_cum_series_streamlit(MAParam, DaysList, CumList):
    with st.expander("Show EWMA Cumulative Series Plot", expanded=False):
        fig = plt.figure('Cum Series plot')
        ax = fig.add_subplot(111)
        for ii in range(len(MAParam)):
            ax.plot(DaysList[ii], CumList[ii], label='MA Param %.2d' % MAParam[ii])
        ax.legend()
        ax.set_xlabel('days')
        ax.set_ylabel('P & L')
        st.pyplot(fig)
        plt.close()

class EWMAout():
    '''
    Class specific to store EWMA data. requires MA parameter to be specified.
    '''
    def __init__(self, MAparam):
        self.model = 'EWMA'
        self.MAparam = MAparam
        self.name = self.model+'%.3d' % self.MAparam
        self.speed = None

    def drop(self, **kwargs):
        '''Attach random variables to this class'''
        for ww in kwargs:
            setattr(self, ww, kwargs[ww])