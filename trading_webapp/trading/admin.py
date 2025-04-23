from django.contrib import admin

# Register your models here.
from .models import Instrument, InstrumentDetails




admin.site.register(Instrument)
admin.site.register(InstrumentDetails)