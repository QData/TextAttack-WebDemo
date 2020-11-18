from django import forms
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse, HttpResponseNotFound
from django.urls import reverse
from django.db.models.functions import Now
from .models import AttackResult

import textattack
import transformers
import uuid
import json
import re

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
    recipe_name = forms.CharField(label='Recipe Name', widget=forms.Select(choices=RECIPE_NAME, attrs={'class' : 'form-control'}))
    

def index(request):
    form = CustomData()
    STORED_POSTS = request.session.get("TextAttackResult")
    if not STORED_POSTS:
        STORED_POSTS = "[]"

    return render(request, 'webdemo/index.html', {'form': form, 'posts': json.loads(STORED_POSTS)})

def attack_interactive(request):
    if request.method == 'POST':
        STORED_POSTS = request.session.get("TextAttackResult")
        form = CustomData(request.POST)
        if form.is_valid():
            input_text, model_name, recipe_name = form.cleaned_data['input_text'], form.cleaned_data['model_name'], form.cleaned_data['recipe_name']
            found = False
            if STORED_POSTS:
                JSON_STORED_POSTS = json.loads(STORED_POSTS)
                for idx, el in enumerate(JSON_STORED_POSTS):
                    if el["input_string"] == input_text:
                        tmp = JSON_STORED_POSTS.pop(idx)
                        JSON_STORED_POSTS.insert(0, tmp)
                        found = True
                        break
                
                if found:
                    request.session["TextAttackResult"] = json.dumps(JSON_STORED_POSTS[:10])
                    return HttpResponseRedirect(reverse('webdemo:index'))

            attack = textattack.commands.attack.attack_args_helpers.parse_attack_from_args(Args(model_name, recipe_name))
            attacked_text = textattack.shared.attacked_text.AttackedText(input_text)
            attack.goal_function.init_attack_example(attacked_text, 1)
            goal_func_result, _ = attack.goal_function.get_result(attacked_text)
            
            input_label = goal_func_result.output
            raw_output = [float(x) for x in list(goal_func_result.raw_output)]
            input_histogram = json.dumps(raw_output)

            result = next(attack.attack_dataset([(input_text, goal_func_result.output)]))
            result_parsed = result.str_lines()
            if len(result_parsed) < 3:
                return HttpResponseNotFound('Failed')
            output_text = result_parsed[2]

            attacked_text_out = textattack.shared.attacked_text.AttackedText(output_text)
            attack.goal_function.init_attack_example(attacked_text_out, 1)
            goal_func_result, _ = attack.goal_function.get_result(attacked_text_out)

            output_label = goal_func_result.output
            raw_output = [float(x) for x in list(goal_func_result.raw_output)]
            output_histogram = json.dumps(raw_output)

            post = {
                        "input_string": input_text, 
                        "model_name": model_name, 
                        "recipe_name": recipe_name, 
                        "output_string": output_text,
                        "input_histogram": input_histogram, 
                        "output_histogram": output_histogram, 
                        "input_label": input_label, 
                        "output_label": output_label
                    }
            
            if STORED_POSTS:
                JSON_STORED_POSTS = json.loads(STORED_POSTS)
                JSON_STORED_POSTS.insert(0, post)
                request.session["TextAttackResult"] = json.dumps(JSON_STORED_POSTS[:10])
            else:
                request.session["TextAttackResult"] = json.dumps([post])

            return HttpResponseRedirect(reverse('webdemo:index'))

        else:
            return HttpResponseNotFound('Failed')

        return HttpResponse('Success')

    return HttpResponseNotFound('<h1>Not Found</h1>')