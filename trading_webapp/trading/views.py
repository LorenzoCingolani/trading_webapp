from django.shortcuts import render

import os
from django.conf import settings
import pandas as pd
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
    


    for file in os.listdir(analysis_input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(analysis_input_folder, file)
            df = pd.read_csv(file_path)
            html_table = df.to_html(classes='table table-bordered', index=False)
            # reove csv from end
            file = file[:-4]
            csvs_dictionary[file] = df.copy()
            


            
    # Create the DataFrame
    control_df_path = os.path.join(settings.BASE_DIR, 'DATA', 'input_main', 'input_main_framework.csv')
    control_df = pd.read_csv(control_df_path)


    
    
    main_analysis(control_df, csvs_dictionary, analysis_input_folder)

    # p2 run validation
    validation_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
    validation_main(control_df, 100, validation_input_folder)
    # p3 run pdm
    pdm_main(control_df, csvs_dictionary)


    # p5 run framework
    # Read products and weights from framewrok input file
    combinedForcast_folder_path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')
    print('combinedForcast_folder_path:', combinedForcast_folder_path)  

    ### Read Portfolio Diversification Multiplier
    PDM=1.86 # copy from 3-PDM_portfolio.h5 file
    aum = 10_000_000
    date_format="%d/%m/%Y"


    # show columns of main file
    print(control_df.columns)
    order_file = framework_main(control_df, combinedForcast_folder_path, csvs_dictionary,  PDM, date_format,aum, is_markov=False)
    output_path = os.path.join(settings.BASE_DIR, 'DATA', 'order_folder', 'orders.csv')
    print(order_file.head())
    order_file.to_csv(output_path, index=False)
    
    

    return render(request, 'trading/run_all.html')
