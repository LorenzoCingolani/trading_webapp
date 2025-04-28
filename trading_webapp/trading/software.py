import pandas as pd
import os
from django.conf import settings

from .strategies import ewma, break_model, carry, save


def main_analysis(framework_df, csvs_dictionary):
    # Strategies and Parameters
	ModelsList = ['ewma01', 'ewma02', 'ewma03', 'ewma04', 'breakout', 'carry']  # Add more if needed
	MAParam = [2, 4, 8, 16]
	BreakParam = [(0.12, 20), (0.16, 20), (0.2, 20), (0.24, 20), (0.28, 20), (0.32, 20)]
	
	# Load data
	print(framework_df.columns)
	for pp in range(framework_df.shape[0]):
		print(print(pp))
		commodity_parameters = framework_df.iloc[pp]
		PrCode= commodity_parameters['Instruments']
		Standard_Cost = commodity_parameters['Standard_Cost']
		exchange_rate = commodity_parameters['Exchange_rate']
		point_value = commodity_parameters['Point_Value']

		print('------------------------------------- Analysing time series for %s'%PrCode)
		print("PrCode",PrCode)
		# check is csv is found
		if not PrCode in csvs_dictionary.keys():
			print(f'No data found for {PrCode} all keys are {csvs_dictionary.keys()}')
			continue
	
		resfold = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
		Params=save.Output('params')
		Params.models=ModelsList
		savecode=PrCode+'_out.h5'
		save.h5file(resfold,savecode,*(Params,))
		data=csvs_dictionary[PrCode].copy()
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

			ResList=ewma.calc(data,MAParam,Standard_Cost,exchange_rate, point_value)
			# save 
			Nres=len(ResList)
			#save.h5file(resfold, savecode, *tuple(ResList)) 

			# cumulative series
			#Params.EWMAbest=MAParam
			#save.h5file(resfold,savecode,*(Params,)) 

			# extract best time-series (move to another file)
			#CumBest=[]
			for ii in range(len(MAParam)):
				for breakout_here in ResList:
					if breakout_here.name=='EWMA%.3d'%MAParam[ii]:   
						StrategyName.append(breakout_here.name) 
						CumList.append(breakout_here.cum_series)
						AvgCapForecastList.append(breakout_here.avg_abs_val_capped_forecast)

			del ResList