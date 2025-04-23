from django.db import models

class Instrument(models.Model):
    instrument_name = models.CharField(max_length=50)
    point_value = models.FloatField()
    tic_size = models.FloatField()
    tic_value = models.FloatField()
    currency = models.CharField(max_length=10)
    exchange_rate = models.FloatField()
    instrument_weights = models.FloatField()
    bid = models.FloatField()
    offer = models.FloatField()
    execution_cost = models.FloatField()
    clearing_cost = models.FloatField()
    standard_cost = models.FloatField()

    def __str__(self):
        return self.instrument_name
    



class InstrumentDetails(models.Model):
    date = models.DateField()
    px_open = models.FloatField(null=True, blank=True)  # Opening price
    px_high = models.FloatField(null=True, blank=True)  # Highest price
    px_low = models.FloatField(null=True, blank=True)  # Lowest price
    px_close_1d = models.FloatField(null=True, blank=True)  # Closing price
    px_volume = models.BigIntegerField(null=True, blank=True)  # Volume traded
    open_int = models.BigIntegerField(null=True, blank=True)  # Open interest
    sectype = models.CharField(max_length=50, null=True, blank=True)  # Security type
    exchange = models.CharField(max_length=50, null=True, blank=True)  # Exchange
    crncy = models.CharField(max_length=10, null=True, blank=True)  # Currency
    tick_size = models.FloatField(null=True, blank=True)  # Tick size
    tick_value = models.FloatField(null=True, blank=True)  # Tick value
    point_value = models.FloatField(null=True, blank=True)  # Point value
    contract_value = models.FloatField(null=True, blank=True)  # Contract value
    name = models.CharField(max_length=100, null=True, blank=True)  # Instrument name
    st_dev = models.FloatField(null=True, blank=True)  # Standard deviation
    no_days = models.IntegerField(null=True, blank=True)  # Number of days since start

    def __str__(self):
        return f"{self.name} - {self.date}"