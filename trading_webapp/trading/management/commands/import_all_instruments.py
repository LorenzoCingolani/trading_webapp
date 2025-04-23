import os
import csv
from django.core.management.base import BaseCommand
from trading.models import InstrumentDetails

class Command(BaseCommand):
    help = 'Import all data from CSV files in the input_instruments folder into the InstrumentDetails model'

    def handle(self, *args, **kwargs):
        folder_path = r'C:\Users\loci_\Desktop\trading_webapp\trading_webapp\DATA\input_instruments'

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path = os.path.join(folder_path, file_name)
                self.stdout.write(f"Processing file: {file_name}")

                with open(file_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Create an InstrumentDetails object for each row
                        InstrumentDetails.objects.create(
                            date=row.get('Date'),
                            px_open=float(row.get('PX_OPEN', 0)) if row.get('PX_OPEN') else None,
                            px_high=float(row.get('PX_HIGH', 0)) if row.get('PX_HIGH') else None,
                            px_low=float(row.get('PX_LOW', 0)) if row.get('PX_LOW') else None,
                            px_close_1d=float(row.get('PX_CLOSE_1D', 0)) if row.get('PX_CLOSE_1D') else None,
                            px_volume=int(row.get('PX_VOLUME', 0)) if row.get('PX_VOLUME') else None,
                            open_int=int(row.get('OPEN_INT', 0)) if row.get('OPEN_INT') else None,
                            sectype=row.get('SECTYPE'),
                            exchange=row.get('EXCHANGE'),
                            crncy=row.get('CRNCY'),
                            tick_size=float(row.get('TICK_SIZE', 0)) if row.get('TICK_SIZE') else None,
                            tick_value=float(row.get('TICK_VALUE', 0)) if row.get('TICK_VALUE') else None,
                            point_value=float(row.get('POINT_VALUE', 0)) if row.get('POINT_VALUE') else None,
                            contract_value=float(row.get('CONTRACT_VALUE', 0)) if row.get('CONTRACT_VALUE') else None,
                            name=row.get('NAME') or row.get('INSTRUMENT'),
                            st_dev=float(row.get('st_dev', 0)) if row.get('st_dev') else None,
                            no_days=int(row.get('no_days', 0)) if row.get('no_days') else None,
                        )
                self.stdout.write(self.style.SUCCESS(f"Data from {file_name} imported successfully!"))

        self.stdout.write(self.style.SUCCESS('All files processed successfully!'))