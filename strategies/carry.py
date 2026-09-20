import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from . import save
from collections import namedtuple
import os
import streamlit as st




def calc(Inst_name,data, exchange_rate=1.0, point_value=50, standard_cost=0.0):

    #data=pd.read_csv(filename)

    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value

    if ('investing_rate' in data.columns) and ('funding_rate' in data.columns):
        hout = carry_foreign(data, standard_cost)
    elif 'far' in data.columns:
        hout = carry_commodity(data, standard_cost)
    else:
        raise NameError(
            'Input file should include eith the column "far" or the columns '+
            '"investing_rate" and "funding_rate"')
    out_csv = os.path.join('DATA', 'output_instruments',f'{Inst_name}_{hout.name}.csv')
    out_plot = os.path.join('DATA', 'output_plots',f'{Inst_name}_{hout.name}.png')

    data.to_csv(out_csv)
    st.write(f"Saving results to {out_csv}")
    # show data with proper headers and size
    st.header(f"Carry Strategy Results for {Inst_name} Rows: {data.shape[0]} Columns: {data.shape[1]} data is")
    st.dataframe(data)
    
    plot_cum_series(np.arange(1,data.shape[0]+1), hout.cum_series, out_plot, Inst_name)

    return hout
  


def carry_foreign(data, standard_cost=0.0):

    ### create output
    hout=CARRYout()
    hout.TH=save.TimeHistory()
    hout.TH.st_dev=np.array(data['st_dev'])
    hout.px_close=np.array(data['near'])
    hout.TH.start_date=data.as_matrloc()[0,0]

    data['stdev_lookback']=36
    stdev_lookback= data['stdev_lookback']
    data['stdev_decay']=2/(stdev_lookback+1)
    data['returns']=data['near'] - data['near'].shift()
    data['sqreturns']=data['returns']*data['returns']

    data['variance'] = 1.#* data['sqreturns']
    for i in range(1,len(data)):
        data.loc[i,'variance']=\
            data.loc[i,'stdev_decay']*data.loc[i,'sqreturns']+\
                                (1-data.loc[i,'stdev_decay'])*data.loc[i-1,'variance']

    data['st_dev']=data['variance']**.5
    data['stdev_yearly']=data['st_dev']*16
    data['price_diff'] = data['investing_rate']-data['funding_rate']
    data['raw_curry'] = data['price_diff'] / data['stdev_yearly']

    avg_abs_val_raw_curry = abs(data['raw_curry']).mean()
   # forecast_scalar = 10. / avg_abs_val_raw_curry
    forecast_scalar = 30

    data['forecast'] = forecast_scalar * data['raw_curry']
    data['capped_forecast']=data['forecast'].clip(-20,+20)


    data['forecast*returns']=data['capped_forecast'].shift()*data['returns']

    # near_pct_change: the raw % price return used inside forecast_pct_return below, saved as
    # its own column for validation (otherwise it's only computed inline and invisible in the
    # output file). 'near' is the instrument's back-adjusted/continuous (i.e. TRADED) price
    # series here, not a raw single-contract quote - it's the same series as PX_CLOSE_1D - so
    # near_pct_change already represents the return actually earned holding the traded/rolled
    # position, matching Carver's distinction between NEARER (used only to build net_exp_ret's
    # curve comparison) and TRADED (the contract performance is actually measured against).
    data['near_pct_change'] = data['near'].pct_change(fill_method=None)

    # forecast_pct_return: consistently-defined percentage-return series used for rule
    # correlations (FDM) and Sharpe calcs across EVERY strategy (Carry and EWMA alike) -
    # forecast*returns above stays on the net_exp_ret/raw-price scale deliberately (preserved
    # for comparison), it is NOT used for correlation/Sharpe. Yesterday's forecast (shift(1),
    # no look-ahead) x today's realized % price return, divided by 10 to match the position-
    # sizing convention (subsystem_pos = vol_scalar * capped_forecast / 10, "forecast of 10" =
    # 1x normal position). Correlation/FDM/Sharpe/Sortino are scale-invariant so the /10 makes
    # no difference there, but it makes Mean Annual Return/Std Dev/Drawdowns/Worst Day-Month
    # truthful - without it they represent a fictitious ~10x-overlevered version of the
    # strategy nobody actually trades. Applied identically to EWMA's forecast_pct_return.
    data['forecast_pct_return'] = (data['capped_forecast'].shift(1) / 10) * data['near_pct_change']

    data.loc[1,'cum_series'] = data.loc[1,'forecast*returns']

    for i in range(2,len(data)):
        data.loc[i,'cum_series']=\
                    data.loc[i,'forecast*returns']+data.loc[i-1,'cum_series']

    #sr=yearly_mean/yearly_stdev
    cum_series_stdev= np.std(data['forecast*returns'])
    cum_series_mean=np.mean(data['forecast*returns'])
    cum_series_sr=cum_series_mean*(math.sqrt(252))/cum_series_stdev
    cum_series=data['cum_series']

    #carry['forecast*return'] = data['capped_forecast']*data['return'].shift(-1)

    aum=10000000
    data['1%_move'] = data['near']*0.01
    # data['point_value']=1/data['near']
    data['block_value']=data['1%_move']*data['point_value']

    data['ICV']=data['st_dev']*data['point_value']
    # data['exchange_rate'] = 1
    data['IVV']=data['ICV']*data['exchange_rate']

    data['Daily_Cash_Vol_Tgt']=aum*.2/16

    data['Volatility_Scalar']=data['Daily_Cash_Vol_Tgt']/data['IVV']

    data['Subsystem_Pos']=(data['Volatility_Scalar']*data['forecast'])/10

    data['Current_pos'] = data['Subsystem_Pos'].shift()
    data['trades_needed']=data['Subsystem_Pos']-data['Current_pos']
    avg_abs_valtgtpos= abs(data['Current_pos']).mean()
    sum_abs_trades_needed= abs(data['trades_needed']).sum()
    years=data.shape[0] / 256
    trades_needed_yearly=sum_abs_trades_needed/years
    turnover=trades_needed_yearly/(2*avg_abs_valtgtpos)

    data['turnover'] = turnover
    data['standard_cost'] = standard_cost

    # saving
    hout.forecast_ret_sr=cum_series_sr
    hout.forecast_scalar=forecast_scalar
    hout.turnover=turnover
    hout.years=years
    hout.cum_series=np.array(cum_series[:])
    # Daily (not cumulative) forecast_pct_return, for cross-strategy correlation/FDM inputs
    # (main_analysis()'s ReturnSeriesList) - correlating cumulative equity curves instead of
    # daily observations produces misleadingly high, unstable correlations.
    hout.daily_forecast_pct_return=np.array(data['forecast_pct_return'])

    return hout


DEFAULT_ROLL_DISTANCE_YEARS = 1 / 12


def roll_distance(data):
    """Years between the near and far contracts = 1 / ROLLS_PER_YEAR (12 monthly, 4 quarterly),
    read from the input file's ROLLS_PER_YEAR column (first row, like the cost columns). Falls back
    to 1/12 when the column is missing/empty/non-positive - which is only right for monthly rolls."""
    if 'ROLLS_PER_YEAR' in data.columns:
        rolls = pd.to_numeric(data['ROLLS_PER_YEAR'], errors='coerce').iloc[0]
        if pd.notna(rolls) and rolls > 0:
            return 1.0 / float(rolls)
    return DEFAULT_ROLL_DISTANCE_YEARS


def carry_commodity(data, standard_cost=0.0):

    ### create output
    hout=CARRYout()
    hout.TH=save.TimeHistory()
    hout.TH.st_dev=np.array(data['st_dev'])
    hout.px_close=np.array(data['near'])
    hout.far=np.array(data['far'])
    hout.TH.start_date=data.Date[0]

    data['stdev_lookback']=36
    stdev_lookback= data['stdev_lookback']
    data['stdev_decay']=2/(stdev_lookback+1)
    # data['returns']=data['near'] - data['near'].shift()
    data['returns']=data['near'].diff()
    data.loc[data.index[0], 'returns'] = 0


    data['sqreturns']=data['returns']*data['returns']
    if 'ROLLS_PER_YEAR' not in data.columns:
        st.warning("ROLLS_PER_YEAR column not found in this instrument's input file - Carry assumes "
                   "monthly rolls (distance 1/12), which is wrong for quarterly-rolling instruments.")
    data['distance']=roll_distance(data)
    valid_prices = (data['near'] != 0) & (data['far'] != 0)
    data['price_diff'] = np.where(valid_prices, data['far'] - data['near'], 0.0)
    data['net_exp_ret']=data['price_diff']/data['distance']

    # Percentage equivalent of price_diff/net_exp_ret, kept as a companion column (same
    # preserve-and-add pattern as forecast_pct_return) - raw_carry/capped_forecast still build
    # from net_exp_ret unchanged, per the earlier deliberate decision to keep the carry signal
    # itself on the raw price basis.
    data['pct_diff'] = np.where(valid_prices, (data['far'] - data['near']) / data['near'], 0.0)
    data['pct_net_exp_ret'] = data['pct_diff'] / data['distance']

    data['stdev_decay']=2/(data['stdev_lookback']+1)
    data.loc[1,'variance'] = data.loc[1,'sqreturns']
    for i in range(2,len(data)):
         data.loc[i,'variance']=data.loc[i,'stdev_decay']*data.loc[i,'sqreturns']+(1-data.loc[i,'stdev_decay'])*data.loc[i-1,'variance']
    data['stdev']=data['variance']**.5
    data['stdev_yearly']=data['stdev']*16 
    data['raw_carry']=data['net_exp_ret']/data['stdev_yearly']
    
    forecast_scalar=30.
    avg_abs_val_raw_curry = abs(data['raw_carry']).mean()
    #forecast_scalar = 10. / avg_abs_val_raw_curry
    data['forecast_scalar']=forecast_scalar

    data['forecast']= data['forecast_scalar']*data['raw_carry']
    data['capped_forecast']=data['forecast'].clip(-20,+20)

    avg_abs_val_capped_forecast_carry = abs(data['capped_forecast']).mean()


    # Today's capped forecast x tomorrow's net expected return (curve-implied carry yield).
    # LEGACY/REFERENCE ONLY: net_exp_ret is a smooth, ~0.9-day-to-day-autocorrelated curve-
    # implied yield level, not an independent daily observation - unsuitable for any
    # correlation/Sharpe/Sortino calculation (that's what forecast_pct_return below is for).
    # Kept unchanged, on this original basis, only because steps/p2_validation.py's
    # REQUIRED_MODEL_COLUMNS uses this column's presence to detect a valid strategy output
    # file. Do not feed this (or its cumulative sum) into any statistic.
    data['forecast*returns'] = data['capped_forecast']*data['net_exp_ret'].shift(-1)

    # near_pct_change: the raw % price return used inside forecast_pct_return below, saved as
    # its own column for validation (otherwise it's only computed inline and invisible in the
    # output file). 'near' is the instrument's back-adjusted/continuous (i.e. TRADED) price
    # series here, not a raw single-contract quote - it's the same series as PX_CLOSE_1D - so
    # near_pct_change already represents the return actually earned holding the traded/rolled
    # position, matching Carver's distinction between NEARER (used only to build net_exp_ret's
    # curve comparison) and TRADED (the contract performance is actually measured against).
    data['near_pct_change'] = data['near'].pct_change(fill_method=None)

    # forecast_pct_return: consistently-defined percentage-return series used for rule
    # correlations (FDM) and Sharpe calcs across EVERY strategy (Carry and EWMA alike) -
    # forecast*returns above stays on the net_exp_ret/raw-price scale deliberately (preserved
    # for comparison), it is NOT used for correlation/Sharpe. Yesterday's forecast (shift(1),
    # no look-ahead) x today's realized % price return, divided by 10 to match the position-
    # sizing convention (subsystem_pos = vol_scalar * capped_forecast / 10, "forecast of 10" =
    # 1x normal position). Correlation/FDM/Sharpe/Sortino are scale-invariant so the /10 makes
    # no difference there, but it makes Mean Annual Return/Std Dev/Drawdowns/Worst Day-Month
    # truthful - without it they represent a fictitious ~10x-overlevered version of the
    # strategy nobody actually trades. Applied identically to EWMA's forecast_pct_return.
    data['forecast_pct_return'] = (data['capped_forecast'].shift(1) / 10) * data['near_pct_change']

    # Forecast Return Shart-Ratio
    forecast_ret_stedv=np.std( data['forecast*returns'][1:-1].values )
    forecast_ret_mean=np.mean( data['forecast*returns'][1:-1].values )
    forecast_ret_sr=forecast_ret_mean*np.sqrt(252)/forecast_ret_stedv 

    #data['cum_series']=data n ['forecast*returns']+data['forecast*returns']
    data.loc[1,'cum_series_carry'] = data.loc[1,'forecast*returns']
    for i in range(2,len(data)):
        data.loc[i,'cum_series_carry']=data.loc[i,'forecast*returns']+data.loc[i-1,'cum_series_carry']

    #sr=yearly_mean/yearly_stdev
    cum_series_stedv_carry= np.std(data['cum_series_carry'])*16
    cum_series_mean_carry=np.mean(data['cum_series_carry'])/2.9087
    cum_series_sr_carry=cum_series_mean_carry/cum_series_stedv_carry
    cum_series_carry=data['cum_series_carry']

    aum=10000000
    data['1%_move'] = data['near']*0.01
    # data['point_value']=1000
    data['block_value']=data['1%_move']*data['point_value']
    data['ICV']=data['st_dev']*data['point_value']
    # data['exchange_rate'] = 1
    data['IVV']=data['ICV']*data['exchange_rate']
    data['Daily_Cash_Vol_Tgt']=aum*.2/16
    data['Volatility_Scalar']=data['Daily_Cash_Vol_Tgt']/data['IVV']
    data['Subsystem_Pos']=data['Volatility_Scalar']*data['capped_forecast']/10
    # answer = str(round(answer, 2))
    # breakout['Tgt_Pos'] =  round(breakout['Subsystem_Pos'], 3)
    data['Target_Pos'] = data['Subsystem_Pos'].round(decimals=0)
    data['Current_pos'] = data['Subsystem_Pos'].shift()
    data['trades_needed']=(data['Subsystem_Pos']-data['Current_pos']).round()


    # Carver's turnover formula: (sum(abs(trades_needed)) / years) / (2 * avg(abs(current_position)))
    carry_avg_abs_val_currentPos = abs(data['Current_pos']).mean() #denominator
    carry_sum_abs_tradesNeeded = abs(data['trades_needed']).sum() #numerator
    years = data.shape[0] / 256
    trades_needed_yearly = carry_sum_abs_tradesNeeded/years
    turnover_carry = trades_needed_yearly/(2*carry_avg_abs_val_currentPos)

    data['turnover'] = turnover_carry
    data['standard_cost'] = standard_cost

    # saving
    hout.avg_abs_val_capped_forecast=avg_abs_val_capped_forecast_carry
    hout.forecast_ret_sr=forecast_ret_sr
    hout.forecast_scalar=forecast_scalar
    hout.turnover=turnover_carry
    hout.years=years
    hout.cum_series=np.array(cum_series_carry[:])
    # Daily (not cumulative) forecast_pct_return, for cross-strategy correlation/FDM inputs
    # (main_analysis()'s ReturnSeriesList) - correlating cumulative equity curves instead of
    # daily observations produces misleadingly high, unstable correlations.
    hout.daily_forecast_pct_return=np.array(data['forecast_pct_return'])

    return hout








# def write_csv(Res,fname):
#     '''
#     write csv
#     '''

#     # open file
#     fout=open(fname,'w')
#     print('Saving into %s' %fname)

#     ### write scalar input
#     fout.write('CARRY'+'\n')

#     fout.write('AvgAbsFor' + ',%.4f'%Res.avg_abs_val_capped_forecast + '\n')
#     fout.write('ForecastReturnSR' + ',%.4f'%Res.forecast_ret_sr + '\n')
#     fout.write('Turnover' +',%.1f'%Res.turnover + '\n')
#     fout.write('ForecastScalar' +',%.4f'%Res.forecast_scalar + '\n')
#     fout.write('Years' + ',%d'%Res.years + '\n')
#     fout.write('CumSeriesLength' + ',%d'%Res.cum_series + '\n')


#     ### write results (array)
#     # check all series have the same length - raise error otherwise
#     Ncum=Res[5]

#     # convert arrays into matrloc (to facilitate writing)
#     M=np.array(Res[6])
#     # write line by line
#     for ii in range(Ncum):
#         fout.write( 'CumSeriesEntry%.5d' %(ii+1) )
#         fout.write(',%.4f'%M[ii] + '\n')
#     fout.close()


class CARRYout():
    '''
    Class specific to store EWMA data. requires MA parameter to be specified.
    '''
    def __init__(self):
        self.model='CARRY'
        self.name=self.model







def plot_cum_series(Days, CumSeries, figname, inst_name):
    with st.expander(f"Show Carry Cumulative Series Plot for {inst_name}", expanded=False):
        fig = plt.figure('Cum Series plot')
        ax = fig.add_subplot(111)
        ax.plot(Days, CumSeries)
        ax.set_xlabel('days')
        ax.set_ylabel('P & L')
        st.pyplot(fig)
        plt.close()
        # Optionally still save the figure if needed
        fig.savefig(figname)
    return



if __name__=='__main__':
    exchange_rate = 1.025
    point_value = 1250
    Res=calc(filename='./Files/all_in_1year/Instruments/ER1.csv', exchange_rate=exchange_rate, point_value=point_value)
