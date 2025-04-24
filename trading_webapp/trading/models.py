from django.db import models

class InstrumentControl(models.Model):
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
    
