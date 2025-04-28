from django.shortcuts import render
from .models import InstrumentControl
import os
from django.conf import settings
import pandas as pd
from .software import main_analysis



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
    # collect data from the database
    csv_folder = os.path.join(settings.BASE_DIR, 'Data', 'input_instruments')
    instruments = InstrumentControl.objects.all()
    control_df = pd.DataFrame(list(instruments.values()))
    csvs_dictionary = {}

    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_folder, file)
            df = pd.read_csv(file_path)
            html_table = df.to_html(classes='table table-bordered', index=False)
            # reove csv from end
            file = file[:-4]
            csvs_dictionary[file] = df.copy()
    main_analysis(control_df, csvs_dictionary)
    
    

    return render(request, 'trading/run_all.html')
