from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns here
    path('', views.home, name='home'),  # Example: Home page
    path('buy/', views.buy, name='buy'),  # Example: Buy page
    path('sell/', views.sell, name='sell'),  # Example: Sell page
]