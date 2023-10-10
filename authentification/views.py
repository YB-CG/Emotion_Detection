from django.shortcuts import render, redirect, HttpResponse
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages

# Create your views here.

def homepage(request):
    return HttpResponse("Hello World")

def register_request(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful")
            return redirect('homepage')
        messages.error(request, "Unsuccessful registration. Invalid information.")
    form = NewUserForm()
    return render (request=request, template_name="register.html", context={"register_form":form})
