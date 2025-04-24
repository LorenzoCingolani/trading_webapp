from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns here
    path('', views.home, name='home'),  # Example: Home page
    path('show_all_data/', views.show_all_data, name='show_all_data'),  # Example: Buy page
    path('sell/', views.sell, name='sell'),  # Example: Sell page
]