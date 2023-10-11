from django.urls import path
from . import views

app_name = 'emotiondetector'  # Add this line to specify the app name

urlpatterns = [
    path("", views.homepage, name="homepage"),
]