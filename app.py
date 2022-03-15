import streamlit as st
import numpy as np

from argparse import Namespace

import logging
import pickle
import textattack
import transformers
import time
import random
import re

from models.html_helper import HtmlHelper
from models.args import Args
from models.cache import Cache

logger = logging.getLogger(__name__)

from config import NUM_SAMPLES_TO_ATTACK, MODELS, ATTACK_RECIPES, HIDDEN_ATTACK_RECIPES, PRECOMPUTED_RESULTS_DICT_NAME, HISTORY

def load_attack(model_name, attack_recipe_name, num_examples):
    # Load model.
    model_class_name = MODELS[model_name][0]
    logger.info(f"Loading transformers.AutoModelForSequenceClassification from '{model_class_name}'.")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_class_name)
    try:
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_class_name)
    except OSError:
        logger.warn('Couldn\'t find tokenizer; defaulting to "bert-base-uncased".')
        tokenizer = textattack.models.tokenizers.AutoTokenizer("bert-base-uncased")
    model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # Load attack.
    logger.info(f"Loading attack from recipe {attack_recipe_name}.")
    attack = eval(f"{ATTACK_RECIPES[attack_recipe_name]}.build(model)")

    # Load dataset.
    _, dataset_args = MODELS[model_name]
    dataset = textattack.datasets.HuggingFaceDataset(
        *dataset_args, shuffle=True
    )
    dataset.examples = dataset.examples[:num_examples]
    return model, attack, dataset

@st.cache
def get_attack_recipe_prototype(attack_recipe_name):
    """ a sort of hacky way to print an attack recipe without loading a big model"""
    recipe = eval(textattack.commands.attack.attack_args.ATTACK_RECIPE_NAMES[attack_recipe_name]).build
    dummy_tokenizer = Namespace(**{ 'encode': None})
    dummy_model = Namespace(**{ 'tokenizer': dummy_tokenizer })
    recipe = recipe(dummy_model)
    recipe_str = str(recipe)
    del recipe
    del dummy_model
    del dummy_tokenizer
    return recipe_str

def display_history(fake_latency=False):
    history = PRECOMPUTE_CACHE.get(HISTORY)
    for idx, result in enumerate(history):
        if fake_latency: random_latency()
        st.markdown(HtmlHelper.get_attack_result_html(idx, result), unsafe_allow_html=True)

def random_latency():
    # Artificially inject a tiny bit of latency to provide
    # a feel of the attack _running_.
    time.sleep(random.triangular(0., 2., .8))

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_and_print_attack_results(model_name, attack_recipe_name, num_examples):
    with st.spinner(f'Loading `{model_name}` model and `{attack_recipe_name}` attack...'):
        model, attack, dataset = load_attack(model_name, attack_recipe_name, num_examples)
    dataset_name = dataset._name

    # Run attack.
    from collections import deque
    worklist = deque(range(0, num_examples))
    results = []
    with st.spinner(f'Running attack on {num_examples} samples from nlp dataset "{dataset_name}"...'):
        for idx, result in enumerate(attack.attack_dataset(dataset, indices=worklist)):
            st.markdown(HtmlHelper.get_attack_result_html(idx, result), unsafe_allow_html=True)
            results.append(result)

    # Update precomputed results
    PRECOMPUTE_CACHE.add((model_name, attack_recipe_name), results)

def run_attack_interactive(text, model_name, attack_recipe_name):
    if PRECOMPUTE_CACHE.exists((text, model_name, attack_recipe_name)) and PRECOMPUTE_CACHE.exists(HISTORY):
        PRECOMPUTE_CACHE.to_top((text, model_name, attack_recipe_name))
        display_history(fake_latency=True)
    else:
        attack = textattack.commands.attack.attack_args_helpers.parse_attack_from_args(Args(model_name, attack_recipe_name))
        attacked_text = textattack.shared.attacked_text.AttackedText(text)
        initial_result = attack.goal_function.get_output(attacked_text)
        result = next(attack.attack_dataset([(text, initial_result)]))

        # Update precomputed results
        PRECOMPUTE_CACHE.add((text, model_name, attack_recipe_name), result)
        display_history()

def run_attack(model_name, attack_recipe_name, num_examples):
    if PRECOMPUTE_CACHE.exists((model_name, attack_recipe_name)):
        PRECOMPUTE_CACHE.to_top((model_name, attack_recipe_name))
        display_history(fake_latency=True)
    else:
        get_and_print_attack_results(model_name, attack_recipe_name, num_examples)
    
def process_attack_recipe_doc(attack_recipe_text):
    attack_recipe_text = attack_recipe_text.strip()
    attack_recipe_text = "\n".join(map(lambda line: line.strip(), attack_recipe_text.split("\n")))
    return attack_recipe_text

def main():
    st.beta_set_page_config(page_title='TextAttack Web Demo', page_icon='https://cdn.shopify.com/s/files/1/1061/1924/products/Octopus_Iphone_Emoji_JPG_large.png', initial_sidebar_state ='auto')
    st.markdown(HtmlHelper.INITIAL_INSTRUCTIONS_HTML, unsafe_allow_html=True)

    # Print TextAttack info to sidebar.
    st.sidebar.markdown('<h1 style="text-align:center; font-size: 1.5em;">TextAttack 🐙</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-size:1.em; text-align:center;"><a href="https://github.com/QData/TextAttack">https://github.com/QData/TextAttack</a></p>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # Select model.
    all_model_names = list(re.sub(r'-mr$', '-rotten_tomatoes', m) for m in MODELS.keys())
    model_names = list(sorted(set(map(lambda x: x.replace(x[x.rfind('-'):],''), all_model_names))))
    model_default = 'bert-base-uncased'
    model_default_index = model_names.index(model_default)
    interactive = st.sidebar.checkbox('Interactive')
    model_name = st.sidebar.selectbox('Model from transformers:', model_names, index=model_default_index)
    
    # Select dataset. (TODO make this less hacky.)
    if interactive:
        interactive_text = st.sidebar.text_input('Custom Input Data')
    matching_model_keys = list(m for m in all_model_names if m.startswith(model_name))
    dataset_names = list(sorted(map(lambda x: x.replace(x[:x.rfind('-')+1],''), matching_model_keys)))
    dataset_default_index = 0
    for optional_dataset_default in ['rotten_tomatoes', 'sst2', 'mnli']:
        try:
            dataset_default_index = dataset_names.index(optional_dataset_default)
            break
        except ValueError:
            continue
    dataset_name = st.sidebar.selectbox('Dataset from nlp:', dataset_names, index=dataset_default_index)
    full_model_name = '-'.join((model_name, dataset_name)).replace('-rotten_tomatoes', '-mr')
    
    # Select attack recipe.
    recipe_names = list(sorted(ATTACK_RECIPES.keys()))
    for hidden_attack in HIDDEN_ATTACK_RECIPES: recipe_names.remove(hidden_attack)
    recipe_default = 'textfooler'
    recipe_default_index = recipe_names.index(recipe_default)
    attack_recipe_name = st.sidebar.selectbox('Attack recipe', recipe_names, index=recipe_default_index)

    # Select number of examples to be displayed
    if not interactive:
        num_examples = st.sidebar.slider('Number of Examples', 1, 100, value=10, step=1)

    # Run attack on button press.
    if st.sidebar.button('Run attack'):
        # Run full attack.
        if interactive: run_attack_interactive(interactive_text, full_model_name, attack_recipe_name)
        else: run_attack(full_model_name, attack_recipe_name, num_examples)
    else:
        # Print History of Usage
        timeline_history = PRECOMPUTE_CACHE.get(HISTORY)
        for idx, entry in enumerate(timeline_history):
            st.markdown(HtmlHelper.get_attack_result_html(idx, entry), unsafe_allow_html=True)

    # Display clear history button
    if PRECOMPUTE_CACHE.exists(HISTORY):
        clear_history = st.button("Clear History")
        if clear_history:
            PRECOMPUTE_CACHE.purge(key=HISTORY)

    # TODO print attack metrics somewhere?
    # Add model info to sidebar.
    hf_model_name = MODELS[full_model_name][0]
    model_link = f"https://huggingface.co/{hf_model_name}"
    st.markdown(f"### Model Hub Link \n [[{hf_model_name}]({model_link})]", unsafe_allow_html=True)
    
    # Add attack info to sidebar (TODO would main page be better?).
    attack_recipe_doc = process_attack_recipe_doc(eval(f"{ATTACK_RECIPES[attack_recipe_name]}.__doc__"))
    st.sidebar.markdown(f'<hr style="margin: 1.0em 0;"> <h3>Attack Recipe:</h3>\n<b>Name:</b> {attack_recipe_name} <br> <br> {attack_recipe_doc}', unsafe_allow_html=True)
    
    # Print attack recipe composition
    attack_recipe_prototype = get_attack_recipe_prototype(attack_recipe_name)
    st.markdown(f'### Attack Recipe Prototype \n```\n{attack_recipe_prototype}\n```')
    
    purge_cache = st.button("Purge Local Cache")
    if purge_cache:
        PRECOMPUTE_CACHE.purge()

if __name__ == "__main__":
    PRECOMPUTE_CACHE = Cache(log=False)
    main()