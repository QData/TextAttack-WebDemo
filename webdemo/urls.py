from django.urls import path

from . import views

app_name = 'webdemo'

urlpatterns = [
    path('', views.index, name='index'),
    path('attack_interactive', views.attack_interactive, name='attack_interactive'),
    path('captum_interactive', views.captum_interactive, name='captum_interactive'),
    path('captum_heatmap_interactive', views.captum_heatmap_interactive, name='captum_heatmap_interactive'),
]