from django.http import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse
from pathlib import Path
import os
from . import model

BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):
	return render(request, os.path.join(BASE_DIR, 'templates','index.html'))

def chat(request):
    context = {}
    context['reply'] = model.evaluateInput(request.GET['messages'])
    return JsonResponse(context)