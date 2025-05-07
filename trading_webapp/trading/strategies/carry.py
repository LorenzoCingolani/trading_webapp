import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from . import save
from collections import namedtuple
import os
from django.conf import settings  # Ensure settings is imported




def calc(Inst_name,data, exchange_rate=1.0, point_value=50):

    #data=pd.read_csv(filename)

    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value

    if ('investing_rate' in data.columns) and ('funding_rate' in data.columns): 
        hout = carry_foreign(data)
    elif 'far' in data.columns:
        hout = carry_commodity(data)
    else:
        raise NameError(
            'Input file should include eith the column "far" or the columns '+
            '"investing_rate" and "funding_rate"')
    out_csv = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments',f'{Inst_name}_{hout.name}.csv')
    out_plot = os.path.join(settings.BASE_DIR, 'DATA', 'output_plots',f'{Inst_name}_{hout.name}.png')

    data.to_csv(out_csv)
    plot_cum_series(np.arange(1,data.shape[0]+1), hout.cum_series, out_plot)

    return hout
  


def carry_foreign(data):

    ### create output
    hout=CARRYout()
    hout.TH=save.TimeHistory()
    hout.TH.st_dev=np.array(data['st_dev'])
    hout.px_close=np.array(data['PX_CLOSE_1D'])
    hout.TH.start_date=data.as_matrloc()[0,0]

    data['stdev_lookback']=36
    stdev_lookback= data['stdev_lookback']
    data['stdev_decay']=2/(stdev_lookback+1)
    data['returns']=data['PX_CLOSE_1D'] - data['PX_CLOSE_1D'].shift()
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
    data['1%_move'] = data['PX_CLOSE_1D']*0.01
    # data['point_value']=1/data['PX_CLOSE_1D']
    data['block_value']=data['1%_move']*data['point_value']

    data['ICV']=data['st_dev']*data['point_value']
    # data['exchange_rate'] = 1
    data['IVV']=data['ICV']/data['exchange_rate']

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

    # saving
    hout.forecast_ret_sr=cum_series_sr
    hout.forecast_scalar=forecast_scalar
    hout.turnover=turnover
    hout.years=years
    hout.cum_series=np.array(cum_series[:])

    return hout


def carry_commodity(data):

    ### create output
    hout=CARRYout()
    hout.TH=save.TimeHistory()
    hout.TH.st_dev=np.array(data['st_dev'])
    hout.px_close=np.array(data['PX_CLOSE_1D'])
    hout.far=np.array(data['far'])
    hout.TH.start_date=data.Date[0]

    data['stdev_lookback']=36
    stdev_lookback= data['stdev_lookback']
    data['stdev_decay']=2/(stdev_lookback+1)
    # data['returns']=data['PX_CLOSE_1D'] - data['PX_CLOSE_1D'].shift()
    data['returns']=data['PX_CLOSE_1D'].diff()
    data.loc[data.index[0], 'returns'] = 0


    data['sqreturns']=data['returns']*data['returns']
    data['price_diff']= data['far'] - data['PX_CLOSE_1D']
    data['distance']=(1/12)
    data['net_exp_ret']=data['price_diff']/data['distance']

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


    # data['forecast*returns'] = data['capped_forecast']*data['returns'].shift(-1)
    data['forecast*returns'] = data['capped_forecast']*data['net_exp_ret'].shift(-1)

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
    data['1%_move'] = data['PX_CLOSE_1D']*0.01
    # data['point_value']=1000
    data['block_value']=data['1%_move']*data['point_value']
    data['ICV']=data['st_dev']*data['point_value']
    # data['exchange_rate'] = 1
    data['IVV']=data['ICV']/data['exchange_rate']
    data['Daily_Cash_Vol_Tgt']=aum*.2/16
    data['Volatility_Scalar']=data['Daily_Cash_Vol_Tgt']/data['IVV']
    data['Subsystem_Pos']=data['Volatility_Scalar']*data['capped_forecast']/10
    # answer = str(round(answer, 2))
    # breakout['Tgt_Pos'] =  round(breakout['Subsystem_Pos'], 3)
    data['Target_Pos'] = data['Subsystem_Pos'].round(decimals=0)
    data['Current_pos'] = data['Subsystem_Pos'].shift()
    data['trades_needed']=(data['Subsystem_Pos']-data['Current_pos']).round()


    #carry_avg_abs_valtgtpos= abs(data['Subsystem_Pos']).mean()
    #sum_abs_trades_needed= abs(data['trades_needed']).sum()
    #years=data['no_days'].dropna().values[-1] / 252
    #carry_trades_needed_yearly=sum_abs_trades_needed/years
    #turnover_carry=carry_trades_needed_yearly/(2*carry_avg_abs_valtgtpos)

     #turnover formula = avg number of trades needed/2*avg abs current pos
    carry_avg_abs_val_currentPos = abs(data['Current_pos']).mean() #denominator
    carry_avg_number_tradesNeeded= abs(data['trades_needed']).mean() #numerator
    years = data.shape[0] / 256
    trades_needed_yearly = carry_avg_number_tradesNeeded/years
    turnover_carry = trades_needed_yearly/(2*carry_avg_abs_val_currentPos)

    # saving
    hout.avg_abs_val_capped_forecast=avg_abs_val_capped_forecast_carry
    hout.forecast_ret_sr=forecast_ret_sr
    hout.forecast_scalar=forecast_scalar
    hout.turnover=turnover_carry
    hout.years=years
    hout.cum_series=np.array(cum_series_carry[:])

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







def plot_cum_series(Days,CumSeries,figname):

    ### plot cumulative series
    fig=plt.figure('Cum Series plot')
    ax=fig.add_subplot(111)
    ax.plot(Days,CumSeries)
    # ax.legend()
    ax.set_xlabel('days')
    ax.set_ylabel('P & L')
    fig.savefig(figname)
    plt.close()

    return



if __name__=='__main__':
    exchange_rate = 1.025
    point_value = 1250
    Res=calc(filename='./Files/all_in_1year/Instruments/ER1.csv', exchange_rate=exchange_rate, point_value=point_value)
