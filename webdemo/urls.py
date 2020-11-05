from django.urls import path

from . import views

app_name = 'webdemo'

urlpatterns = [
    path('', views.index, name='index'),
    path('attack_interactive', views.attack_interactive, name='attack_interactive')
]