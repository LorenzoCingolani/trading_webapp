# filepath: c:\Users\loci_\Desktop\trading_webapp\trading_webapp\trading\management\commands\import_csv.py
import csv
from django.core.management.base import BaseCommand
from trading.models import Instrument

class Command(BaseCommand):
    help = 'Import data from CSV into the Instrument model'

    def handle(self, *args, **kwargs):
        file_path = r'C:\Users\loci_\Desktop\trading_webapp\trading_webapp\DATA\input_main_framework.csv'
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                Instrument.objects.create(
                    instrument_name=row['Instruments'],
                    point_value=float(row['Point Value']),
                    tic_size=float(row['Tic Size']),
                    tic_value=float(row['Tic Value']),
                    currency=row['Currency'],
                    exchange_rate=float(row['Exchange rate']),
                    instrument_weights=float(row['Instrument_Weights']),
                    bid=float(row['bid']),
                    offer=float(row['offer']),
                    execution_cost=float(row['execution_cost']),
                    clearing_cost=float(row['clearing_cost']),
                    standard_cost=float(row['Standard Cost']),
                )
        self.stdout.write(self.style.SUCCESS('Data imported successfully!'))