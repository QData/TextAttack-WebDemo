from django.urls import path

from . import views

app_name = 'webdemo'

urlpatterns = [
    path('', views.index, name='index'),
    path('interactive_attack', views.interactive_attack, name='interactive_attack'),
]