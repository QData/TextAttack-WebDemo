import streamlit as st

from argparse import Namespace

import logging
import pickle
import textattack
import transformers
import time
import random
import re

logger = logging.getLogger(__name__)

INITIAL_INSTRUCTIONS_HTML = """<p style="font-size:1.em; font-weight: 300">👋 Welcome to the TextAttack demo app! Please select a model and an attack recipe from the dropdown.</p> <hr style="margin: 1.em 0;">"""

from config import NUM_SAMPLES_TO_ATTACK, MODELS, ATTACK_RECIPES, HIDDEN_ATTACK_RECIPES, PRECOMPUTED_RESULTS_DICT_NAME

def load_precomputed_results():
    try:
        precomputed_results = pickle.load(open(PRECOMPUTED_RESULTS_DICT_NAME, "rb" ))
    except FileNotFoundError:
        precomputed_results = {}
    print(f'Found {len(precomputed_results)} keys in pre-computed results.')
    return precomputed_results

def load_attack(model_name, attack_recipe_name):
    # Load model.
    model_class_name = MODELS[model_name][0]
    logger.info(f"Loading transformers.AutoModelForSequenceClassification from '{model_class_name}'.")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_class_name)
    try:
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_class_name)
    except OSError:
        logger.warn('Couldn\'t find tokenizer; defaulting to "bert-base-uncased".')
        tokenizer = textattack.models.tokenizers.AutoTokenizer("bert-base-uncased")
    setattr(model, "tokenizer", tokenizer)
    # Load attack.
    logger.info(f"Loading attack from recipe {attack_recipe_name}.")
    attack = eval(f"{ATTACK_RECIPES[attack_recipe_name]}(model)")
    # Load dataset.
    _, dataset_args = MODELS[model_name]
    dataset = textattack.datasets.HuggingFaceNlpDataset(
        *dataset_args, shuffle=True
    )
    return model, attack, dataset

def improve_result_html(result_html):
    result_html = result_html.replace("color = bold", 'style="font-weight: bold;"')
    result_html = result_html.replace("color = underline", 'style="text-decoration: underline;"')
    result_html = result_html.replace('<font style="font-weight: bold;"', '<span style=""') # no bolding for now
    result_html = result_html.replace('<font style="text-decoration: underline;"', '<span style="text-decoration: underline;"')
    result_html = re.sub(r"<font\scolor\s=\s(\w.*?)>", r'<span style="background-color: \1; padding: 1.2px; font-weight: 600;">', result_html)
    # replace font colors with transparent highlight versions
    result_html = result_html.replace(': red', ': rgba(255, 0, 0, .7)') \
                             .replace(': green', ': rgb(0, 255, 0, .7)') \
                             .replace(': blue', ': rgb(0, 0, 255, .7)') \
                             .replace(': gray', ': rgb(220, 220, 220, .7)')
    result_html = result_html.replace("</font>", "</span>")
    return result_html

def get_attack_result_status(attack_result):
    status_html = attack_result.goal_function_result_str(color_method='html')
    return improve_result_html(status_html)
    
def get_attack_result_html(idx, attack_result):
    result_status = get_attack_result_status(attack_result)
    result_html_lines = attack_result.str_lines(color_method='html')
    result_html_lines = [improve_result_html(line) for line in result_html_lines]
    rows = [
        ['', result_status],
        ['Input', result_html_lines[1]]
    ]
    
    if len(result_html_lines) > 2:
        rows.append(['Output', result_html_lines[2]])
    
    table_html = '\n'.join((f'<b>{header}:</b> {body} <br>' if header else f'{body} <br>') for header,body in rows)
    return f'<h3>Result {idx+1}</h3> {table_html} <br>'

@st.cache
def get_attack_recipe_prototype(attack_recipe_name):
    """ a sort of hacky way to print an attack recipe without loading a big model"""
    recipe = eval(textattack.commands.attack.attack_args.ATTACK_RECIPE_NAMES[attack_recipe_name])
    dummy_tokenizer = Namespace(**{ 'encode': None})
    dummy_model = Namespace(**{ 'tokenizer': dummy_tokenizer })
    recipe = recipe(dummy_model)
    recipe_str = str(recipe)
    del recipe
    del dummy_model
    del dummy_tokenizer
    return recipe_str

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_and_print_attack_results(model_name, attack_recipe_name):
    with st.spinner(f'Loading `{model_name}` model and `{attack_recipe_name}` attack...'):
        model, attack, dataset = load_attack(model_name, attack_recipe_name)
    dataset_name = dataset._name
    # Run attack.
    from collections import deque
    worklist = deque(range(0, NUM_SAMPLES_TO_ATTACK))
    results = []
    with st.spinner(f'Running attack on {NUM_SAMPLES_TO_ATTACK} samples from nlp dataset "{dataset_name}"...'):
        for idx, result in enumerate(attack.attack_dataset(dataset, indices=worklist)):
            st.markdown(get_attack_result_html(idx, result), unsafe_allow_html=True)
            results.append(result)
    
    # Update precomputed results
    PRECOMPUTED_RESULTS = load_precomputed_results()
    PRECOMPUTED_RESULTS[(model_name, attack_recipe_name)] = results
    pickle.dump(PRECOMPUTED_RESULTS, open(PRECOMPUTED_RESULTS_DICT_NAME, 'wb'))
    # Return results
    return { 'results': results, 'already_printed': True }

def random_latency():
    # Artificially inject a tiny bit of latency to provide
    # a feel of the attack _running_.
    time.sleep(random.triangular(0., 2., .8))

def run_attack(model_name, attack_recipe_name):
    if (model_name, attack_recipe_name) in PRECOMPUTED_RESULTS:
        results = PRECOMPUTED_RESULTS[(model_name, attack_recipe_name)]
        for idx, result in enumerate(results):
            random_latency()
            st.markdown(get_attack_result_html(idx, result), unsafe_allow_html=True)
    else:
        # Precompute results
        results_dict = get_and_print_attack_results(model_name, attack_recipe_name)
        results = results_dict['results']
        # Print attack results, as long as this wasn't the first time they were computed.
        if not results_dict['already_printed']:
            for idx, result in enumerate(results):
                random_latency()
                st.markdown(get_attack_result_html(idx, result), unsafe_allow_html=True)
        results_dict['already_printed'] = False
    # print summary
    

def process_attack_recipe_doc(attack_recipe_text):
    attack_recipe_text = attack_recipe_text.strip()
    attack_recipe_text = "\n".join(map(lambda line: line.strip(), attack_recipe_text.split("\n")))
    return attack_recipe_text

def main():
    # Print instructions.
    st.markdown(INITIAL_INSTRUCTIONS_HTML, unsafe_allow_html=True)
    # Print TextAttack info to sidebar.
    st.sidebar.markdown('<h1 style="text-align:center; font-size: 1.5em;">TextAttack 🐙</h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-size:1.em; text-align:center;"><a href="https://github.com/QData/TextAttack">https://github.com/QData/TextAttack</a></p>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    # Select model.
    all_model_names = list(re.sub(r'-mr$', '-rotten_tomatoes', m) for m in MODELS.keys())
    model_names = list(sorted(set(map(lambda x: x.replace(x[x.rfind('-'):],''), all_model_names))))
    model_default = 'bert-base-uncased'
    model_default_index = model_names.index(model_default)
    model_name = st.sidebar.selectbox('Model from transformers:', model_names, index=model_default_index)
    # Select dataset. (TODO make this less hacky.)
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
    # Run attack on button press.
    if st.sidebar.button('Run attack'):
        # Run full attack.
        run_attack(full_model_name, attack_recipe_name)
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

if __name__ == "__main__": # @TODO split model & dataset into 2 dropdowns
    PRECOMPUTED_RESULTS = load_precomputed_results()
    main()