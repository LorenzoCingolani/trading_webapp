from django.shortcuts import render
from .models import Instrument, InstrumentDetails


def home(request):
    return render(request, 'trading/home.html')


def buy(request):
    return render(request, 'trading/buy.html')



def sell(request):
    return render(request, 'trading/sell.html')
