from django.shortcuts import render

import os
from django.conf import settings
import pandas as pd
import json
from .p1_analysis import main_analysis
from .p2_validation import validation_main
from .p3_pdm import pdm_main
from .p5_framework import framework_main



def home(request):
    return render(request, 'trading/home.html')
    



def show_all_data(request):
    csv_folder = os.path.join(settings.BASE_DIR, 'Data', 'input_instruments')
    csv_data = []

    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            df = pd.read_csv(file_path)
            html_table = df.to_html(classes='table table-bordered', index=False)
            csv_data.append({'filename': file, 'table': html_table})

    

    return render(request, 'trading/show_all_data.html', {'csv_data': csv_data})



def run_all(request):
    # p1 collect data from the database
    analysis_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'input_instruments')
    
    csvs_dictionary = {}
    
    # Load the JSON file into the control variable
    json_file_path = os.path.join(settings.BASE_DIR, 'DATA', 'input_main', 'input_main.json')
    with open(json_file_path, 'r') as file:
        control = json.load(file)

    #columsn i need 'SECTYPE', 'EXCHANGE', 'CRNCY', 'TICK_SIZE',
     #  'TICK_VALUE', 'POINT_VALUE', 'CONTRACT_VALUE', 'near', 'far', 'st_dev',
     #  'no_days'

    
    for file in os.listdir(analysis_input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(analysis_input_folder, file)
            df = pd.read_csv(file_path)
            
            # reove csv from end
            file = file[:-4]
            csvs_dictionary[file] = df.copy()
            # update control variable with the current intrument
            if file in control:
                control[file]['INSTRUMENT'] = file
                control[file]['CURRENCY'] = df['CRNCY'].iloc[0]
                control[file]['EXCHANGE'] = df['EXCHANGE'].iloc[0]
                control[file]['SECTYPE'] = df['SECTYPE'].iloc[0]
                control[file]['TICK_SIZE'] = df['TICK_SIZE'].iloc[0]
                control[file]['TICK_VALUE'] = df['TICK_VALUE'].iloc[0]
                control[file]['POINT_VALUE'] = df['POINT_VALUE'].iloc[0]
                control[file]['CONTRACT_VALUE'] = df['CONTRACT_VALUE'].iloc[0]

            

    


    
    main_analysis(control, csvs_dictionary, analysis_input_folder)

    # p2 run validation
    validation_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
    inst_names = list(csvs_dictionary.keys())
    validation_main(inst_names,control, 100, validation_input_folder)
    # p3 run pdm
    pdm_main(control, csvs_dictionary)


    # p5 run framework
    # Read products and weights from framewrok input file
    combinedForcast_folder_path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')
    print('combinedForcast_folder_path:', combinedForcast_folder_path)  

    ### Read Portfolio Diversification Multiplier
    PDM=1.86 # copy from 3-PDM_portfolio.h5 file
    aum = 10_000_000
    date_format="%d/%m/%Y"


    # show columns of main file
    order_file = framework_main(control, combinedForcast_folder_path, csvs_dictionary,  PDM, date_format,aum, is_markov=False)
    output_path = os.path.join(settings.BASE_DIR, 'DATA', 'order_folder', 'orders.csv')
    print(order_file.head())
    order_file.to_csv(output_path, index=False)
    
    

    return render(request, 'trading/run_all.html')
