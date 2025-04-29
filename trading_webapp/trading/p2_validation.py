
import os

import pandas as pd
import numpy as np

from datetime import timedelta, datetime


def load_commodity_data(commodity,CsvFolder):
	all_data = {}
	all_output_files = os.listdir(CsvFolder)
	print('all output file',all_output_files)
	for filename in all_output_files:
		if filename.startswith(commodity) and filename.endswith('.csv'):
			print(filename)
			data = pd.read_csv(os.path.join(CsvFolder, filename))

			data.dropna(subset=['Date'],inplace=True)
			try:
				data['Date'] = pd.to_datetime(data['Date'],  format="%d/%m/%Y")
			except ValueError:
				data['Date'] = pd.to_datetime(data['Date'],  format='mixed', dayfirst=True )
			
			all_data[filename.replace(f'{commodity}_', '').replace('.csv','')] =  data
		else:
			print(f"File {filename} does not match the commodity {commodity}.")
            
	return all_data


def forecast(commodity_data, Weights):
	CumList = [data['forecast*returns'].values for data in commodity_data]
	####### Correlation Matrix
	CorrMat = pd.DataFrame(CumList).T.corr()
	#M= 1/SQRT(W x H x WT)
	M= min(1./np.sqrt(np.dot(Weights.T,np.dot(CorrMat,Weights))), 2.5) #THIS IS THE ORIGINAL
	CapForecastList = [data['capped_forecast'].iloc[-1] for data in commodity_data]
	#CHANGED!!!!!!!!!!!!!!!!!
	#AvgCapForecastList = [data['capped_forecast'].iloc[-1] if not data['capped_forecast'].empty else 0 for data in commodity_data]	
	### Forecast - combined forecast in outer framework
	#FinalForecast = ((w1*f1)+(w2*f2)+(w3*f3)+(w4*f4))*M
	UnweightedForecast= np.dot( Weights,CapForecastList)
	FinalForecast=M*UnweightedForecast
	return FinalForecast, M


def validation_main(framework_df, validation_days, CsvFolder):
	# To have start date which is fixed
	# start_date = datetime(2020, 1, 4)
	for ind, commodity_parameters in framework_df.iterrows():
		PrCode= commodity_parameters['Instruments']
		print(f"Computing forecasts for {PrCode}")
		
		commodity_data = load_commodity_data(PrCode,CsvFolder)

		# 05/10/2009
		
		print(f"commodity_data is {commodity_data}")
		NModels = len(commodity_data)
		#if NModels == 0:
		#	print(f"No models found for {PrCode}")
		#	continue
		print(f"number of models: {NModels}")
		try:
			Weights = 1/NModels * np.ones((NModels))
		except ZeroDivisionError:
			print(f"No models found for {PrCode}")
			continue
		

		# general output
		model1 = list(commodity_data.values())[0]
		# To have start  date from end date till x(validation_days)
		if validation_days == -1:
			start_date = model1['Date'].iloc[1]
		else:
			start_date = model1['Date'].max() - timedelta(days=validation_days)

		val_days = model1[model1['Date'] >=start_date]['Date']
		validation_data = []
		for days in val_days.tolist():
			commodity_subset = [data[data['Date']<days] for data in commodity_data.values()]
			forecasted_value = forecast(commodity_subset, Weights)
			validation_data.append((days, *forecasted_value))

		# create validation DF
		output = pd.DataFrame(validation_data, columns=['Date', 'FinalForecast', 'Multiplier'])
		# save forecast with all models 
		for key, data in commodity_data.items():
			output[f'{key}_forecast'] = data[data['Date']>=start_date]['capped_forecast'].values

		output.to_csv(os.path.join(CsvFolder,'..', 'combinedForecast',f'{PrCode}.csv'),index=False)

if __name__ == '__main__':
	#controls

	MAIN_FILE = 'input_framework_port - Original - Test.csv'
	MAIN_Folder = 'portfolio1'

	# Paths
	framework_input_file=f'./Files/{MAIN_FILE}'
	#CsvFolder='./Files/all_out_1year/portfolioSmall'
	CsvFolder=f'./Files/all_out_1year/{MAIN_Folder}'
 
	#CsvFolder='./Files/all_in_1year/portfolio1'

	validation_days = 100
	validation_date = datetime(2023, 9, 17)

	framework_df = pd.read_csv(framework_input_file)
	#all_output_files = os.listdir(CsvFolder)

	validation_main(framework_df, validation_days, CsvFolder)
