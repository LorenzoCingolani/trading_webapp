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

def run_all(request):
    def stream_logs():
        # p1 collect data from the database
        yield "Collecting data from the database...\n"
        analysis_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'input_instruments')
        
        csvs_dictionary = {}
        
        # Load the JSON file into the control variable
        json_file_path = os.path.join(settings.BASE_DIR, 'DATA', 'input_main', 'input_main.json')
        with open(json_file_path, 'r') as file:
            control = json.load(file)
        yield "Loaded JSON control file.\n"

        # Process CSV files
        for file in os.listdir(analysis_input_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(analysis_input_folder, file)
                df = pd.read_csv(file_path)
                yield f"Processing file: {file}\n"
                
                # Remove .csv from the end
                file = file[:-4]
                csvs_dictionary[file] = df.copy()
                
                # Update control variable with the current instrument
                if file in control:
                    control[file]['INSTRUMENT'] = file
                    control[file]['CURRENCY'] = df['CRNCY'].iloc[0]
                    control[file]['EXCHANGE'] = df['EXCHANGE'].iloc[0]
                    control[file]['SECTYPE'] = df['SECTYPE'].iloc[0]
                    control[file]['TICK_SIZE'] = df['TICK_SIZE'].iloc[0]
                    control[file]['TICK_VALUE'] = df['TICK_VALUE'].iloc[0]
                    control[file]['POINT_VALUE'] = df['POINT_VALUE'].iloc[0]
                    control[file]['CONTRACT_VALUE'] = df['CONTRACT_VALUE'].iloc[0]
                    yield f"Updated control for instrument: {file}\n"

        # p1 main analysis
        yield "Running main analysis...\n"
        main_analysis(control, csvs_dictionary, analysis_input_folder)

        # p2 run validation
        yield "Running validation...\n"
        validation_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
        inst_names = list(csvs_dictionary.keys())
        validation_main(inst_names, control, 100, validation_input_folder)

        # p3 run pdm
        yield "Running PDM...\n"
        pdm = pdm_main(control, csvs_dictionary)
        yield 

        # p5 run framework
        yield "Running framework...\n"
        combinedForcast_folder_path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')

        # Read Portfolio Diversification Multiplier
        PDM = 1.86  # Copy from 3-PDM_portfolio.h5 file
        aum = 10_000_000
        date_format = "%d/%m/%Y"

        # Show columns of main file
        order_file = framework_main(control, combinedForcast_folder_path, csvs_dictionary, PDM, date_format, aum, is_markov=False)
        output_path = os.path.join(settings.BASE_DIR, 'DATA', 'order_folder', 'orders.csv')
        yield "Generated order file:\n"
        yield order_file.head().to_html(classes='table table-bordered', index=False) + "\n"
        order_file.to_csv(output_path)
        yield f"Order file saved to: {output_path}\n"
        # download the file


    # Return a streaming response
    return StreamingHttpResponse(stream_logs(), content_type='text/event-stream') 