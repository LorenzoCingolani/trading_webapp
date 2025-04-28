import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from . import save

from collections import namedtuple





def calc(filename,LookBackList):

    db=pd.read_csv(filename)
    pxclose=db['PX_CLOSE_1D']

    # # clean data: remove NaN
    # iivec=np.isnan(pxclose)==False
    # pxclose=pxclose[iivec]
    # db=db[iivec]

    Return=pxclose.diff()
    Return[0]=0.0
    ReturnSq=Return**2
    Ndays=db.shape[0]

    # saving
    # Res=[]
    Hlist=[]
    CumSeriesList=[]

    TH=save.TimeHistory()
    TH.px_close=pxclose


    for lookback in LookBackList:


        hout=BREAKout(lookback)
        hout.TH=TH


        print('lookback %d' %lookback)
        # define strategy: 1: buy, -1: sell
        Strategy=np.zeros((Ndays,))

        for ii in range(lookback,Ndays):

            ### Define strategy
            MaxRange=np.max(pxclose[ii-lookback:ii])
            MinRange=np.min(pxclose[ii-lookback:ii])

            if pxclose[ii]>MaxRange:
                Strategy[ii]=1 # add smoothing??
            elif pxclose[ii]<MinRange:
                Strategy[ii]=-1
            else:
                Strategy[ii]=Strategy[ii-1]

        ### Compute Variance and related
        StdDevDecay=2./37.
        Variance=np.zeros((Ndays,))
        Variance[:lookback+1]=ReturnSq[:lookback+1]
        for nn in range(lookback+1,Ndays):
            Variance[nn]=StdDevDecay*ReturnSq[nn]+(1.-StdDevDecay)*Variance[nn-1]

        ### Forecast Scalar
        VolatAdj=Return/np.sqrt(Variance)
        # remove nan values
        iiNaN=np.isnan(VolatAdj)#-1*(np.isnan(Variance)-1)
        VolatAdj[iiNaN]=0.0
        ForecastScalar=10./np.average(np.abs(VolatAdj))
        Forecast=VolatAdj*ForecastScalar

        ### rounding and capping
        ForecastCap=np.round(Forecast)
        iilow=ForecastCap<-20.0
        ForecastCap[iilow]=-20.0
        iiup=ForecastCap>20.0
        ForecastCap[iiup]=20.0
        ForecastCap_avg_abs=np.mean(np.abs(ForecastCap))

        ### forecast return and cum series
        ForecastReturn=np.zeros((Ndays,))
        CumSeries=np.zeros((Ndays,))
        for nn in range(lookback+1,Ndays):
            ForecastReturn[nn]=Return[nn]*ForecastCap[nn-1]
            CumSeries[nn]=CumSeries[nn-1]+ForecastReturn[nn]
           
        ### Forecast return sharp-ratio
        ForecastReturn_stdev=np.std( ForecastReturn )
        ForecastReturn_mean=np.mean( ForecastReturn )
        ForecastReturn_sr=ForecastReturn_mean*np.sqrt(252)/ForecastReturn_stdev   


        ###################################### turnover

        BlockValue=0.01*pxclose
        db['point_value']=50
        ICV=db['st_dev']*db['point_value']
        db['exchange_rate']=1
        IVV=ICV/db['exchange_rate']                   

        aum=1000000
        DailyCashVolTgt=aum*.2/16
        VolatilityScalar=DailyCashVolTgt/IVV
        Subsystem_Pos=VolatilityScalar*ForecastCap/10.
        PotfolioInstrPos=Subsystem_Pos
        TargetPos=np.round(PotfolioInstrPos)
        
        CurrentPos=np.zeros((Ndays,))
        TradesNeeded=np.zeros((Ndays,))
        for nn in range(lookback+1,Ndays):
            TradesNeeded[nn]=TargetPos[nn]-CurrentPos[nn-1]
            CurrentPos[nn]=CurrentPos[nn-1]+TradesNeeded[nn]

        Nyears=Ndays/252
        Turnover=np.sum(np.abs(TradesNeeded))/\
                       (Nyears*2.*np.average(np.abs(TargetPos)))

        # saving
        hout.forecast_x_return=ForecastReturn
        hout.avg_abs_val_capped_forecast=ForecastCap_avg_abs
        hout.forecast_ret_sr=ForecastReturn_sr
        hout.forecast_scalar=ForecastScalar
        hout.turnover=Turnover
        hout.cum_series=CumSeries#np.array(cum_series_carry[1:-10])
        Hlist.append( hout )

        # # saving
        # ResHere=[]
        # ResHere.append(ForecastCap_avg_abs) # avg absolute capped forecast
        # ResHere.append(ForecastReturn_sr)         # cum sharp ratio
        # ResHere.append(Turnover)                  # turnover
        # ResHere.append(ForecastScalar)            # Forcast scalar not required
        # ResHere.append(Nyears)                       # years
        # ResHere.append(len(CumSeries))     # lunghezza cum series        
        # ResHere.append(CumSeries)      # cum series
        # ResHere.append(lookback)
        CumSeriesList.append(CumSeries)
        # Res.append(ResHere)

    # Plot all cumulative series
    figname=filename.replace('_in/','_out/')
    figname=figname.replace('.csv','.png')
    plot_cum_series(np.linspace(0,Nyears,Ndays),CumSeriesList,LookBackList,figname)


    # # save h5 file
    # cout=save.Output()
    # self.product=None
    # self.name='break-in'
    # self.strategy='break-in'
    # self.CumSeriesSR=CumSeriesList
    # self.AvgAbsVal=None
    # self.ForecastScalar=Fore
    # self.turnover=None
    # self.Years=None
    # self.CumSeriesTH=None
    # self.ForecastReturnTH=None


    return Hlist
  

# def write_csv(ResAll,froot):
#     '''write csv'''


#     Nlookback=len(ResAll)
#     for ii in range(Nlookback):

#         Res=ResAll[ii]

#         if froot[-4:]=='.csv':
#             fname=froot[:-4]+'_lkbk%.3d'%Res[-1]+'.csv'
#         else:
#             fname=froot+'_lkbk%.3d'%Res[-1]

#         # open file
#         fout=open(fname,'w')
#         print('Saving into %s' %fname)

#         ### write scalar input
#         fout.write('Breakout'+'\n')

#         fout.write('AvgAbsFor' + ',%.4f'%Res[0] + '\n')
#         fout.write('ForecastReturnSR' + ',%.4f'%Res[1] + '\n')
#         fout.write('Turnover' +',%.1f'%Res[2] + '\n')
#         fout.write('ForecastScalar' +',%.4f'%Res[3] + '\n')
#         fout.write('Years' + ',%d'%Res[4] + '\n')
#         fout.write('CumSeriesLength' + ',%d'%Res[5] + '\n')

#         ### write results (array)
#         # check all series have the same length - raise error otherwise
#         Ncum=Res[5]

#         # convert arrays into matrix (to facilitate writing)
#         M=np.array(Res[6])
#         # write line by line
#         for ii in range(Ncum):
#             fout.write( 'CumSeriesEntry%.5d' %(ii+1) )
#             fout.write(',%.4f'%M[ii] + '\n')
#         fout.close()


def plot_cum_series(Days,CumSeriesList,LookBackList,figname):

    #plt_cl=['r','k','b','0.6']

    ### plot cumulative series
    fig=plt.figure('Cum Series plot',(10,6))
    ax=fig.add_subplot(111)
    
    for cc in range(len(LookBackList)):
        ax.plot(Days,CumSeriesList[cc],label='lookback %.2d' %LookBackList[cc])

    ax.legend()
    ax.set_xlabel('years')
    ax.set_ylabel('P & L')
    fig.savefig(figname)
    plt.close()

    return


class BREAKout():
    '''
    Class specific to store EWMA data. requires MA parameter to be specified.
    '''

    def __init__(self,lookback):
        self.model='BREAK'
        self.lookback=lookback
        self.name=self.model+'%.3d'%self.lookback




if __name__=='__main__':

    filename='/home/smub/Documents/teaching/Lorenzo/src/Files/break_in/break_tyc.csv'
    

    LookBackList=[20,80]
    calc(filename,LookBackList)


