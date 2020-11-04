from django import forms
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

import textattack
import transformers
import json
import re

from .models import AttackResult
from .config import MODELS, ATTACK_RECIPES, HIDDEN_ATTACK_RECIPES

MODEL_NAMES = [(x, x) for x in list(re.sub(r'-mr$', '-rotten_tomatoes', m) for m in MODELS.keys())]
RECIPE_NAME = [(x, x) for x in list(sorted(ATTACK_RECIPES.keys()))]

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
    input_text = forms.CharField(label='Custom Data', widget=forms.TextInput(attrs={'class' : 'form-control customDataInput'}))
    model_name = forms.CharField(label='Model Name', widget=forms.Select(choices=MODEL_NAMES, attrs={'class' : 'form-control'}))
    recipe_name = forms.CharField(label='Model Name', widget=forms.Select(choices=RECIPE_NAME, attrs={'class' : 'form-control'}))
    

def index(request):
    if request.method == 'POST':
        form = CustomData(request.POST)
        if form.is_valid():
            input_text, model_name, recipe_name = form.cleaned_data['input_text'], form.cleaned_data['model_name'], form.cleaned_data['recipe_name']
            if AttackResult.objects.filter(input_string=input_text, model_name=model_name, recipe_name=recipe_name).exists():
                db_object = AttackResult.objects.get(input_string=input_text, model_name=model_name, recipe_name=recipe_name)
                result = db_object.output_string
                model_name = db_object.model_name
                recipe_name = db_object.recipe_name
            else:
                attack = textattack.commands.attack.attack_args_helpers.parse_attack_from_args(Args(model_name, recipe_name))
                attacked_text = textattack.shared.attacked_text.AttackedText(input_text)
                attack.goal_function.init_attack_example(attacked_text, 1)
                goal_func_result, _ = attack.goal_function.get_result(attacked_text)
                
                input_label = goal_func_result.output
                raw_output = [float(x) for x in list(goal_func_result.raw_output)]
                input_histogram = json.dumps(raw_output)

                result = next(attack.attack_dataset([(input_text, goal_func_result.output)]))
                result_parsed = result.str_lines()
                print("-"*100)
                print(result_parsed)
                print("-"*100)
                if len(result_parsed) < 3:
                    HttpResponseRedirect(reverse('webdemo:index'))
                    return
                output_text = result_parsed[2]

                attacked_text_out = textattack.shared.attacked_text.AttackedText(output_text)
                attack.goal_function.init_attack_example(attacked_text_out, 1)
                goal_func_result, _ = attack.goal_function.get_result(attacked_text_out)

                output_label = goal_func_result.output
                raw_output = [float(x) for x in list(goal_func_result.raw_output)]
                output_histogram = json.dumps(raw_output)

                AttackResult.objects.update_or_create(session_id=request.session._session_key, input_string=input_text, model_name=model_name, recipe_name=recipe_name, output_string=output_text, input_histogram=input_histogram, output_histogram=output_histogram, input_label=input_label, output_label=output_label)
        else:
            HttpResponseRedirect(reverse('webdemo:index'))

        form = CustomData()
        posts = AttackResult.objects.filter(input_string=input_text, model_name=model_name, recipe_name=recipe_name)
        return render(request, 'webdemo/index.html', {'form': form, 'posts': posts})
    else:
        form = CustomData()
        posts = AttackResult.objects.filter(session_id=request.session._session_key)[:10]

        return render(request, 'webdemo/index.html', {'form': form, 'posts': posts})
    