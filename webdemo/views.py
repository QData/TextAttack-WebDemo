from django import forms
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

import textattack
import transformers
import re

from .models import AttackResult
from .config import MODELS, ATTACK_RECIPES, HIDDEN_ATTACK_RECIPES

MODEL_NAMES = [(x, x) for x in list(re.sub(r'-mr$', '-rotten_tomatoes', m) for m in MODELS.keys())]
RECIPE_NAME = [(x, x) for x in list(sorted(ATTACK_RECIPES.keys()))]

improvements = {
    "color = bold": 'style="font-weight: bold;"',
    "color = underline": 'style="text-decoration: underline;"',
    '<font style="font-weight: bold;"': '<span style=""', # no bolding for now
    '<font style="text-decoration: underline;"': '<span style="text-decoration: underline;"',
    "</font>": "</span>",

    # colors
    ': red': ': rgba(255, 0, 0, .7)',
    ': green': ': rgb(0, 255, 0, .7)',
    ': blue': ': rgb(0, 0, 255, .7)',
    ': gray': ': rgb(220, 220, 220, .7)',

    # icons
    "-->": "<span>&#8594;</span>",
    "1 (": "<span>&#128522;</span> (",
    "0 (": "<span>&#128543;</span> (",
}

class Args():
    def __init__(self, model, recipe, model_batch_size=32, query_budget=200, model_cache_size=2**18, constraint_cache_size=2**18):
        self.model = model
        self.recipe = recipe
        self.model_batch_size = model_batch_size
        self.model_cache_size = model_cache_size
        self.query_budget = query_budget
        self.constraint_cache_size = constraint_cache_size

    def __getattr__(self, item):
        return False

class CustomData(forms.Form):
    model_name = forms.CharField(label='Model Name', widget=forms.Select(choices=MODEL_NAMES, attrs={'class' : 'form-control'}))
    recipe_name = forms.CharField(label='Model Name', widget=forms.Select(choices=RECIPE_NAME, attrs={'class' : 'form-control'}))
    input_text = forms.CharField(label='Custom Data', widget=forms.TextInput(attrs={'class' : 'form-control customDataInput'}))

def index(request):
    form = CustomData()

    return render(request, 'webdemo/index.html', {'form': form, 'results': ['No results to show.']})
    

def interactive_attack(request):
    if request.method == 'POST':
        form = CustomData(request.POST)
        if form.is_valid():
            input_text, model_name, recipe_name = form.cleaned_data['input_text'], form.cleaned_data['model_name'], form.cleaned_data['recipe_name']
            if AttackResult.objects.filter(input_string=input_text, attack_type='-'.join([model_name, recipe_name]  )).exists():
                result = AttackResult.objects.get(input_string=input_text, attack_type='-'.join([model_name, recipe_name])).output_string
            else:
                attack = textattack.commands.attack.attack_args_helpers.parse_attack_from_args(Args(model_name, recipe_name))
                attacked_text = textattack.shared.attacked_text.AttackedText(input_text)
                initial_result = attack.goal_function.get_output(attacked_text)
                result = next(attack.attack_dataset([(input_text, initial_result)]))

                AttackResult.objects.update_or_create(input_string=input_text, attack_type='-'.join([model_name, recipe_name]), output_string=result)
        else:
            HttpResponseRedirect(reverse('webdemo:index'))

    form = CustomData()
    return render(request, 'webdemo/index.html', {'form': form, 'results': [result]*10})