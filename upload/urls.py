from django.urls import path
from . import views

app_name = 'upload'

urlpatterns = [
    path('', views.upload, name='upload'),
    path('upload_image/', views.upload_image, name='upload_image'),
]