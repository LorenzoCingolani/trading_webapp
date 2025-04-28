import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import save
from django.conf import settings
import os



def calculate_forecast_for_breakout(
    adjusted_price: pd.Series, horizon: int = 10
) -> pd.Series:

    max_price = adjusted_price.rolling(horizon, min_periods=1).max()
    min_price = adjusted_price.rolling(horizon, min_periods=1).min()
    mean_price = (max_price + min_price) / 2
    raw_forecast = 40 * (adjusted_price - mean_price) / (max_price - min_price)
    smoothed_forecast = raw_forecast.ewm(span=int(np.ceil(horizon / 4))).mean()

    return smoothed_forecast


def calc_turnover(data, aum=10000000):

    data['1%_move'] = data['PX_CLOSE_1D']*0.01
    # data['point_value']=data['point_value']#50
    data['block_value'] = data['1%_move']*data['point_value']
    data['price_volatility'] = (data['st_dev']/data['PX_CLOSE_1D'] )
    data['ICV'] = data['price_volatility'] * data['block_value']
    # data['exchange_rate']=1
    data['IVV'] = data['ICV'] * data['exchange_rate']
    # asssume 20% exp return based on back test ...so [(aum x 20%) divided by 16 to make it daily]
    data['Daily_Cash_Vol_Tgt'] = aum*.2/16
    data['Volatility_Scalar'] = data['Daily_Cash_Vol_Tgt']/data['IVV']

    data['Subsystem_Pos'] = data['Volatility_Scalar'] * data['capped_forecast']/10.
    #data['Portfolio_instrument_pos'] = data['Subsystem_pos']*PDM*fm['Instrument_Weights']
    
    data['Target_Pos'] =  data['Subsystem_Pos'].round(decimals=0)
    data['Current_pos'] = data['Target_Pos'].shift()

    data['trades_needed']=data['Target_Pos']-data['Current_pos']

    avg_abs_valtgtpos= abs(data['Current_pos']).mean()
    print(f"avg_abs_valtgtpos is {avg_abs_valtgtpos}")
    sum_abs_trades_needed= abs(data['trades_needed']).sum()
    print(f"sum_abs_trades_needed is {sum_abs_trades_needed}")
    years=data['no_days'].dropna().values[-1] / 252
    trades_needed_yearly=sum_abs_trades_needed/years
    turnover=trades_needed_yearly/(2*avg_abs_valtgtpos)
    print(f"turnover is {turnover}")

    return turnover


def calc(Inst_name,data, MAParam, standard_cost, exchange_rate=1.0, point_value=50):


    data['exchange_rate'] = exchange_rate
    data['point_value'] = point_value
    breakouts = []
    CumList = []   # list to store cumulative series (plot only, not saving)
    DaysList = []  # list to store days (for plot only)

    for scalar, horizon in MAParam:

        # create output
        Breakouthere = Brekoutout(scalar, horizon)
        Breakouthere.TH = save.TimeHistory()
        Breakouthere.TH.px_close = np.array(data['PX_CLOSE_1D'])
        Breakouthere.TH.st_dev = np.array(data['st_dev'])
        Breakouthere.TH.start_date = data.values[0, 0]

        print("Calculating breakout %d" % scalar)
        data['Daily_Return'] = data['PX_CLOSE_1D'].pct_change()
        
        data['returns'] = data['PX_CLOSE_1D'].diff()
        data.loc[0, 'returns'] = 0.

        data['sqreturns'] = data['returns']*data['returns']
        
        data['forecast_scalar'] = scalar
        # for testing passing closing value, but we need to pass Adjusted values
        data['smoothed_forecast'] = calculate_forecast_for_breakout(data['PX_CLOSE_1D'], horizon)
        data['forecast'] = data['smoothed_forecast']*scalar
        data['capped_forecast'] = data['forecast'] # .clip(-20, +20) # currently not clipped as we do min-max scaling
        avg_abs_val_capped_forecast = abs(data['capped_forecast']).mean()

        ##################

        # calculate turnover
        turnover = calc_turnover(data)  # <------- to be floced !!!!

        data['abs_forecast'] = abs(data['forecast'])

        data['forecast*returns'] = data['capped_forecast']*data['Daily_Return'].shift(-1)

        data['cum_series'] = data['forecast*returns'].cumsum()
        # embed()
        # Forecast Return Shart-Ratio
        breakout_gross_ret_stedv = np.std(data['forecast*returns'])
        breakout_gross_ret_mean = np.mean(data['forecast*returns'])
        breakout_gross_ret_sr = breakout_gross_ret_mean*np.sqrt(256)/breakout_gross_ret_stedv
        
        if breakout_gross_ret_sr > 0.:
            breakout_net_ret_sr = breakout_gross_ret_sr-(breakout_gross_ret_sr*standard_cost)
        else:
            breakout_net_ret_sr = breakout_gross_ret_sr + (breakout_gross_ret_sr*standard_cost)   

        years = data['no_days'].dropna().values[-1] / 256

        Breakouthere.standard_cost = standard_cost
        Breakouthere.avg_abs_val_capped_forecast = avg_abs_val_capped_forecast
        Breakouthere.breakout_gross_ret_sr = breakout_gross_ret_sr
        Breakouthere.breakout_net_ret_sr = breakout_net_ret_sr
        Breakouthere.turnover = turnover
        Breakouthere.forecast_scalar = scalar
        Breakouthere.years = years
        Breakouthere.breakout_gross_ret_sr  = breakout_gross_ret_sr 
        Breakouthere.cum_series = np.array(data.cum_series.copy()[:])  # cum series
        breakouts.append(Breakouthere)

        # store variables for plotting
        CumList.append(data['cum_series'])
        DaysList.append(data['no_days'])

    # Plot all cumulative series 
    plot_file = os.path.join(settings.BASE_DIR, 'Data', 'output_plots', f'{Inst_name}_stochastic_breakout.png')
    plot_cum_series(MAParam, DaysList, CumList, plot_file)

    return breakouts


def plot_cum_series(MAParam, DaysList, CumList, figname):

    # plot cumulative series
    fig = plt.figure('Cum Series plot')
    ax = fig.add_subplot(111)
    for ii in range(len(MAParam)):
        ax.plot(DaysList[ii], CumList[ii], label=f'MA Param  {MAParam[ii]}')
    ax.legend()
    ax.set_xlabel('days')
    ax.set_ylabel('P & L')
    fig.savefig(figname)
    plt.close()

    return


class Brekoutout():
    '''
    Class specific to store Breakout data. requires MA parameter to be specified.
    '''
    def __init__(self, scalar, horizon):
        self.model = 'Stochastic Breakout'
        self.scalar = scalar
        self.horizon = horizon
        self.name = self.model + '%.3f_%.3d'%(self.scalar, self.horizon)

    def drop(self, **kwargs):
        '''Attach random variables to this class'''
        for ww in kwargs:
            setattr(self, ww, kwargs[ww])


if __name__ == '__main__':
    filename = './Files/all_in_1year/instrument_updated_close/rx1 comdty.csv'
    standard_cost = 0.002
    BreakParam = [(0.12, 20)]
    breakout_params = [()]
    exchange_rate = 1.0
    point_value = 1250
    Res = calc(filename, BreakParam, standard_cost, exchange_rate, point_value)
