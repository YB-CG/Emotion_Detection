from django.urls import path
from . import views

app_name = 'upload'

urlpatterns = [
    path('', views.upload, name='upload'),
    # path('after/', views.after, name='after'),
]