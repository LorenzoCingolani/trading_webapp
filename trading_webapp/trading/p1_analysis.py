import pandas as pd
import os
import numpy as np
from django.conf import settings


from .strategies import ewma, break_model, carry, save, stoch
from .strategies import stochastic_breakout as breakout



def main_analysis(framework_df, csvs_dictionary, MainFolderPath):
    # Strategies and Parameters
	ModelsList = ['ewma01', 'ewma02', 'ewma03', 'ewma04', 'breakout', 'carry']  # Add more if needed
	MAParam = [2, 4, 8, 16]
	BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]
	
	# Load data
	print(framework_df)
	for ins_name in csvs_dictionary.keys():
		print('Instrument name:', ins_name)
		commodity_parameters = framework_df[ins_name]
		Inst_name= commodity_parameters['INSTRUMENT']
		Standard_Cost = commodity_parameters['STANDARD_COST']
		exchange_rate = commodity_parameters['EXCHANGE_RATE']
		point_value = commodity_parameters['POINT_VALUE']

		print('------------------------------------- Analysing time series for %s'%Inst_name)
		print("Inst_name",Inst_name)
		# check is csv is found
		if not Inst_name in csvs_dictionary.keys():
			print(f'No data found for {Inst_name} all keys are {csvs_dictionary.keys()}')
			continue
	
		resfold = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
		Params=save.Output('params')
		Params.models=ModelsList
		savecode=Inst_name+'_out.h5'
		save.h5file(resfold,savecode,*(Params,))
		data=csvs_dictionary[Inst_name].copy()
		##### Allocate memory for variables common to each model
		print(data.columns)

		StrategyName=[] # different than ModelsList - has extra details (e.g. lookback etc)
		CumList=[]
		AvgCapForecastList=[]

		############################################### EWMA

		# Loop through the different lookback periods

		if 'ewma01' in ModelsList and 'ewma03' in ModelsList:
			# compute 
			print('Inside EWMA')

			ResList=ewma.calc(Inst_name,data,MAParam,Standard_Cost,exchange_rate, point_value)
			# save 
			Nres=len(ResList)
			save.h5file(resfold, savecode, *tuple(ResList)) 

			# cumulative series
			#Params.EWMAbest=MAParam
			save.h5file(resfold,savecode,*(Params,)) 

			# extract best time-series (move to another file)
			#CumBest=[]
			for ii in range(len(MAParam)):
				for breakout_here in ResList:
					if breakout_here.name=='EWMA%.3d'%MAParam[ii]:   
						StrategyName.append(breakout_here.name) 
						CumList.append(breakout_here.cum_series)
						AvgCapForecastList.append(breakout_here.avg_abs_val_capped_forecast)

			del ResList


		if 'breakout' in ModelsList:
			# compute 
			print('Inside Breakout')

			ResList=breakout.calc(Inst_name,data,BreakParam,Standard_Cost,exchange_rate, point_value)
			# save 
			Nres=len(ResList)
			save.h5file(resfold, savecode, *tuple(ResList)) 

			# cumulative series
			Params.BreakoutBest=BreakParam
			save.h5file(resfold,savecode,*(Params,)) 

			# extract best time-series (move to another file)
			#CumBest=[]
			for ii in range(len(BreakParam)):
				for breakout_here in ResList:
					if breakout_here.name=='Stochastic Breakout%.3f_%.3d'%(BreakParam[ii][0],BreakParam[ii][1]):

						StrategyName.append(breakout_here.name) 
						CumList.append(breakout_here.cum_series)
						AvgCapForecastList.append(breakout_here.avg_abs_val_capped_forecast)

			del ResList

		############################################### CARRY

		if 'carry' in ModelsList and 'far' in data.columns.tolist():
			Res=carry.calc(Inst_name,data, exchange_rate, point_value)
			save.h5file(resfold,savecode,*(Res,))

			StrategyName.append(Res.name)
			CumList.append(Res.cum_series)
			AvgCapForecastList.append(Res.avg_abs_val_capped_forecast)		
			del Res

		############################################### Stock

		if 'stoch_in' in ModelsList:

			# compute 
			Res=stoch.calc(Inst_name,data)
			save.h5file(resfold,savecode,*(Res,))
			# print output 
			output_name=os.path.abspath(MainFolderPath)+'/stoch_out/stoch_%s.csv'%Inst_name
			#stoch.write_csv(Res,output_name)
			# cumulative 
			StrategyName.append(Res.name)
			CumList.append(Res.cum_series)
			AvgCapForecastList.append(Res.avg_abs_val_capped_forecast)
			del Res

		############################################## Break-out

		if 'break01' in ModelsList and 'break03' in ModelsList:
			# compute 
			ResList=break_model.calc(Inst_name,data,LookBackList)
			Nres=len(ResList)
			save.h5file(resfold,savecode,*tuple(ResList))
			# # print output 
			# output_name=os.path.abspath(MainFolderPath)+'/break_out/break_%s.csv'%Inst_name
			# break_model.write_csv(Res,output_name)

			# cumulative series and avg cap forecST
			for cc in range(Nres):
				StrategyName.append(ResList[cc].name)
				CumList.append(ResList[cc].cum_series)
				AvgCapForecastList.append(ResList[cc].avg_abs_val_capped_forecast)
			del ResList
		
		NModels = len(AvgCapForecastList)

		CorrMat = pd.DataFrame(CumList).T.corr()
		
		Weights = 1/NModels * np.ones((NModels))
		# ### Assign weights to strategies
		# 50% to carry
		# equally distributed the others
		
		#Weights = 1 #THIS IS TO BE COMMENTED OUT FOR ON PRODUCT ONLY ON EWMA
		#Weights[ModelsList.index('carry')]=0.5

		# random weigths
		# Weights =np.random.rand(NModels)
		# Weights =Weights/np.linalg.norm(Weights,ord=1)

		### Multiplier
		# 1/sqrt(W * H * WT) capped at 2.5(250%)
		M= min(1./np.sqrt(np.dot(Weights.T,np.dot(CorrMat,Weights))), 2.5) #THIS IS THE ORIGINAL

		### Forecast - combined forecast in outer framework
		#FinalForecast = ((w1*f1)+(w2*f2)+(w3*f3)+(w4*f4))*M
		UnweightedForecast=np.sum( np.dot( Weights,AvgCapForecastList ))
		FinalForecast=M*UnweightedForecast

		# general output
		Forecast=save.Output('forecast')
		Forecast.strategies=StrategyName
		Forecast.strategy_weight=Weights
		Forecast.multiplier=M
		Forecast.unweighted_forecast=UnweightedForecast
		Forecast.weighted_forecast=FinalForecast
		Forecast.CorrelationMatrix = CorrMat
		
		save.h5file(resfold,savecode,*(Forecast,))
