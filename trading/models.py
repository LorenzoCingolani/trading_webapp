from django.db import models

class InstrumentControl(models.Model):
    Instruments = models.CharField(max_length=50)
    Point_Value = models.FloatField()
    Tic_Size = models.FloatField()
    Tic_Value = models.FloatField()
    Currency = models.CharField(max_length=10)
    Exchange_rate = models.FloatField()
    Instrument_Weights = models.FloatField()
    bid = models.FloatField()
    offer = models.FloatField()
    execution_cost = models.FloatField()
    clearing_cost = models.FloatField()
    Standard_Cost = models.FloatField()

    def __str__(self):
        return self.Instruments
