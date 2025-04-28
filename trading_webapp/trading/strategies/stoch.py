import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from . import save

from collections import namedtuple



def calc(filename):

    data=pd.read_csv(filename)


    hout=STOCHout()
    hout.TH=save.TimeHistory()
    hout.TH.px_close=np.array(data['PX_CLOSE_1D'])
    hout.TH.st_dev=np.array(data['st_dev'])
    hout.px_low=np.array(data['PX_LOW'])
    hout.px_high=np.array(data['PX_HIGH'])
    hout.TH.start_date=data.as_matrix()[0,0]

    # data['returns']=data['PX_CLOSE_1D']-data['PX_CLOSE_1D'].shift()
    data['returns']=data['PX_CLOSE_1D'].diff()
    data.returns.iloc[0] = 0.

    data['roll_mean']=(data['PX_HIGH']+data['PX_LOW'])/2

    delta = (data['PX_HIGH']-data['PX_LOW'])
    data['scalare']=(data['PX_CLOSE_1D']-data['roll_mean'])/delta

    cap = .01*data['PX_HIGH'].mean()
    iivec = np.where( delta.abs().values<cap )[0]
    for ii in iivec:
        if ii>0:
            data.scalare[ii] = data.scalare[ii-1] 
        else:
            data.scalare[ii] = 1.


    data['forecast']=data['scalare']*40

    avg_abs_val_capped_forecast_breakout = abs(data['forecast']).mean()

    data['forecast*returns'] = data['forecast']*data['returns'].shift(-1)
    data.ix[0,'cum_series_breakout'] = data.ix[0,'forecast*returns']
    for i in range(1,len(data)): 
        data.ix[i,'cum_series_breakout']=data.ix[i,'forecast*returns']+data.ix[i-1,'cum_series_breakout'] 
    cum_series_breakout=data['cum_series_breakout']

    years=data['no_days'].dropna().values[-1] / 252    
    
    # Forecast Return Shart-Ratio
    forecast_ret_stedv=np.std( data['forecast*returns'] )
    forecast_ret_mean=np.mean( data['forecast*returns'] )
    forecast_ret_sr=forecast_ret_mean*np.sqrt(252)/forecast_ret_stedv 

    #turnover
    aum=1000000
    data['1%_move'] = data['PX_CLOSE_1D']*0.01
    data['point_value']=50
    data['block_value']=data['1%_move']*data['point_value']
    data['ICV']=data['st_dev']*data['point_value']
    data['exchange_rate'] = 1
    data['IVV']=data['ICV']/data['exchange_rate']
    data['Daily_Cash_Vol_Tgt']=aum*.2/16
    data['Volatility_Scalar']=data['Daily_Cash_Vol_Tgt']/data['IVV']
    data['Subsystem_Pos']=data['Volatility_Scalar']*data['forecast']/10
    #answer = str(round(answer, 2))
    #data['Tgt_Pos'] =  round(data['Subsystem_Pos'], 3)
    data['Current_pos'] = data['Subsystem_Pos'].shift()
    data['trades_needed']=data['Subsystem_Pos']-data['Current_pos']
    avg_abs_valtgtpos= abs(data['Subsystem_Pos']).mean()
    sum_abs_trades_needed= abs(data['trades_needed']).sum()

    trades_needed_yearly=sum_abs_trades_needed/years
    turnover_breakout=trades_needed_yearly/(2*avg_abs_valtgtpos)

    #data['Current_pos'].fillna(0).head()
    #dataframe[column].fillna(0)
    #data['trades_needed'].fillna(data['Current_pos']).head()


    # saving
    hout.forecast_x_return=np.array(data['forecast*returns'])
    hout.avg_abs_val_capped_forecast=avg_abs_val_capped_forecast_breakout
    hout.forecast_ret_sr=forecast_ret_sr
    hout.forecast_scalar=None#forecast_scalar
    hout.turnover=turnover_breakout
    hout.years=years
    hout.cum_series=np.array(cum_series_breakout[:])

    # ResHere=[]
    # ResHere.append(avg_abs_val_capped_forecast_breakout) # avg absolute capped forecast
    # ResHere.append(forecast_ret_sr)         # cum sharp ratio
    # ResHere.append(turnover_breakout)		# turnover
    # ResHere.append(None)                    # Forcast scalar not required
    # ResHere.append(years)						 # years
    # ResHere.append(len(cum_series_breakout)-11)     # lunghezza cum series        
    # ResHere.append(cum_series_breakout[1:-10])      # cum series

    # Plot all cumulative series
    figname=filename.replace('_in/','_out/')
    figname=figname.replace('.csv','.png')
    plot_cum_series(data['no_days'],data['cum_series_breakout'],figname)


    return hout
  



# def write_csv(Res,fname):
#     '''
#     write csv
#     '''

#     # open file
#     fout=open(fname,'w')
#     print('Saving into %s' %fname)

#     ### write scalar input
#     fout.write('Stochastic'+'\n')

#     fout.write('AvgAbsFor' + ',%.4f'%Res[0] + '\n')
#     fout.write('ForecastReturnSR' + ',%.4f'%Res[1] + '\n')
#     fout.write('Turnover' +',%.1f'%Res[2] + '\n')
#     #fout.write('ForecastScalar' +',%.4f'%Res[3] + '\n')
#     fout.write('ForecastScalar' +',NA' + '\n')
#     fout.write('Years' + ',%d'%Res[4] + '\n')

#     fout.write('CumSeriesLength' + ',%d'%Res[5] + '\n')


#     ### write results (array)
#     # check all series have the same length - raise error otherwise
#     Ncum=Res[5]

#     # convert arrays into matrix (to facilitate writing)
#     M=np.array(Res[6])
#     # write line by line
#     for ii in range(Ncum):
#         fout.write( 'CumSeriesEntry%.5d' %(ii+1) )
#         fout.write(',%.4f'%M[ii] + '\n')
#     fout.close()


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


class STOCHout():
    '''
    Class specific to store EWMA data. requires MA parameter to be specified.
    '''
    def __init__(self):
        self.model='STOCH'
        self.name=self.model


# -----------------------------------------------------------

if __name__=='__main__':
    filename='./Files/all_in/ED7.csv'
    Res=calc(filename)