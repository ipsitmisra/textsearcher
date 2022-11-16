from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('scan', views.scan , name='scan'),
    path('upload_file', views.upload_file , name='upload_file'),
]
