from django.shortcuts import render
from django.http import StreamingHttpResponse
import time


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
from django.http import StreamingHttpResponse
import time

def single_line():
    return '-' * 50 + '\n'



def run_all(request):
    analysis_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'input_instruments')
    csvs_dictionary = {}

    json_file_path = os.path.join(settings.BASE_DIR, 'DATA', 'input_main', 'input_main.json')
    with open(json_file_path, 'r') as file:
        control = json.load(file)

    for file in os.listdir(analysis_input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(analysis_input_folder, file)
            df = pd.read_csv(file_path)
            csvs_dictionary[file[:-4]] = df.copy()

            if file[:-4] in control:
                name = file[:-4]
                control[name].update({
                    'INSTRUMENT': name,
                    'CURRENCY': df['CRNCY'].iloc[0],
                    'EXCHANGE': df['EXCHANGE'].iloc[0],
                    'SECTYPE': df['SECTYPE'].iloc[0],
                    'TICK_SIZE': df['TICK_SIZE'].iloc[0],
                    'TICK_VALUE': df['TICK_VALUE'].iloc[0],
                    'POINT_VALUE': df['POINT_VALUE'].iloc[0],
                    'CONTRACT_VALUE': df['CONTRACT_VALUE'].iloc[0],
                    'EXCHANGE_RATE': df['Exchange rate'].iloc[0],
                    'STANDARD_COST': df['Standard Cost'].iloc[0],
                })

    print('main_analysis')
    single_line()
    main_analysis(control, csvs_dictionary, analysis_input_folder)
    print('main_analysis done')


    single_line()
    validation_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
    inst_names = list(csvs_dictionary.keys())
    validation_main(inst_names, control, 100, validation_input_folder)
    print('validation_main done')
    single_line()

    pdm = pdm_main(control, csvs_dictionary)
    print('pdm_main done')

    single_line()
    combinedForcast_folder_path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')

    PDM = 1.86  # Copy from 3-PDM_portfolio.h5 file
    aum = 10_000_000
    date_format = "%d/%m/%Y"

    order_file = framework_main(control, combinedForcast_folder_path, csvs_dictionary, PDM, date_format, aum, is_markov=False)
    output_path = os.path.join(settings.BASE_DIR, 'DATA', 'order_folder', 'orders.csv')
    order_file.to_csv(output_path)

    single_line()
    single_line()

    return render(request, 'trading/run_all.html', {'order_file': order_file.to_html(classes='table table-bordered', index=False)})