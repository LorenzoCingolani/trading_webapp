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
        yield "data: Collecting data from database...\n\n"
        analysis_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'input_instruments')
        csvs_dictionary = {}

        json_file_path = os.path.join(settings.BASE_DIR, 'DATA', 'input_main', 'input_main.json')
        with open(json_file_path, 'r') as file:
            control = json.load(file)
        yield "data: Loaded JSON control file.\n\n"

        for file in os.listdir(analysis_input_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(analysis_input_folder, file)
                df = pd.read_csv(file_path)
                csvs_dictionary[file[:-4]] = df.copy()
                yield f"data: Loaded file: {file}\n\n"

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
                    })
                    yield f"data: Updated control for {name}\n\n"

        # main_analysis
        for log in main_analysis(control, csvs_dictionary, analysis_input_folder):
            yield f"data: {log}\n\n"

        # validation_main
        validation_input_folder = os.path.join(settings.BASE_DIR, 'DATA', 'output_instruments')
        for log in validation_main(list(csvs_dictionary.keys()), control, 100, validation_input_folder):
            yield f"data: {log}\n\n"

        # pdm_main
        for log in pdm_main(control, csvs_dictionary):
            yield f"data: {log}\n\n"

        # framework_main
        combined_path = os.path.join(settings.BASE_DIR, 'DATA', 'combinedForecast')
        PDM = 1.86
        aum = 10_000_000
        date_format = "%d/%m/%Y"

        for log in framework_main(control, combined_path, csvs_dictionary, PDM, date_format, aum, is_markov=False):
            if isinstance(log, dict) and log.get("type") == "order_file":
                html = log["data"].head().to_html(classes='table table-bordered', index=False)
                yield f"data: <br><b>Order File:</b><br>{html}\n\n"
                output_path = os.path.join(settings.BASE_DIR, 'DATA', 'order_folder', 'orders.csv')
                log["data"].to_csv(output_path, index=False)
                yield f"data: Order file saved to: {output_path}\n\n"
            else:
                yield f"data: {log}\n\n"

    return StreamingHttpResponse(stream_logs(), content_type='text/event-stream')
