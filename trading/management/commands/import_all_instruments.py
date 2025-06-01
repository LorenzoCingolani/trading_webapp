import os
import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from trading.models import InstrumentData

class Command(BaseCommand):
    help = 'Import all data from CSV files in the input_instruments folder into the InstrumentData model'

    def handle(self, *args, **kwargs):
        folder_path = r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\trading_webapp\DATA\input_instruments'

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path = os.path.join(folder_path, file_name)
                self.stdout.write(f"Processing file: {file_name}")

                with open(file_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Parse and convert the date format
                        raw_date = row.get('Date')
                        try:
                            parsed_date = datetime.strptime(raw_date, '%d/%m/%Y').date()  # Adjust format as needed
                        except ValueError:
                            self.stdout.write(self.style.ERROR(f"Invalid date format: {raw_date}"))
                            continue

                        # Create an InstrumentData object for each row
                        InstrumentData.objects.create(
                            date=parsed_date,
                            px_open=float(row.get('PX_OPEN', 0)) if row.get('PX_OPEN') else 0.0,
                            px_high=float(row.get('PX_HIGH', 0)) if row.get('PX_HIGH') else 0.0,
                            px_low=float(row.get('PX_LOW', 0)) if row.get('PX_LOW') else 0.0,
                            px_close_1d=float(row.get('PX_CLOSE_1D', 0)) if row.get('PX_CLOSE_1D') else 0.0,
                            px_volume=int(row.get('PX_VOLUME', 0)) if row.get('PX_VOLUME') else 0,
                            open_int=int(row.get('OPEN_INT', 0)) if row.get('OPEN_INT') else 0,
                            instrument=row.get('INSTRUMENT', 'not_defined'),
                            sectype=row.get('SECTYPE', 'not_defined'),
                            exchange=row.get('EXCHANGE', 'not_defined'),
                            crncy=row.get('CRNCY', 'not_defined'),
                            tick_size=float(row.get('TICK_SIZE', 0)) if row.get('TICK_SIZE') else 0.0,
                            tick_value=float(row.get('TICK_VALUE', 0)) if row.get('TICK_VALUE') else 0.0,
                            point_value=float(row.get('POINT_VALUE', 0)) if row.get('POINT_VALUE') else 0.0,
                            contract_value=float(row.get('CONTRACT_VALUE', 0)) if row.get('CONTRACT_VALUE') else 0.0,
                            near=float(row.get('near', 0)) if row.get('near') else 0.0,
                            far=float(row.get('far', 0)) if row.get('far') else 0.0,
                            st_dev=float(row.get('st_dev', 0)) if row.get('st_dev') else 0.0,
                            no_days=int(row.get('no_days', 0)) if row.get('no_days') else 0,
                        )
                self.stdout.write(self.style.SUCCESS(f"Data from {file_name} imported successfully!"))

        self.stdout.write(self.style.SUCCESS('All files processed successfully!'))