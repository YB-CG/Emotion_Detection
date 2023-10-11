from django.urls import path
from . import views
app_name = 'live'  # Add this line to specify the app name

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
