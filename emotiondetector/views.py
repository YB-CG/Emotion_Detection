from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required(login_url='login')  # Replace 'login' with the actual URL name of your login page
def homepage(request):
    return render(request, 'homepage.html')
