import argparse
import hashlib
import os
import pickle
import shlex
import textattack
import torch

from config import NUM_SAMPLES_TO_ATTACK, PRECOMPUTED_RESULTS_DICT_NAME

def parse_ta_args(args_str):
    parser = argparse.ArgumentParser(
        "TextAttack CLI",
        usage="[python -m] texattack <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="textattack command helpers")
    
    # Register commands
    textattack.commands.attack.AttackCommand.register_subcommand(subparsers)
    
    args = parser.parse_args(shlex.split(args_str))
    
    # set defaults
    args.checkpoint_resume = False
    args.disable_stdout = True
    args.num_examples = NUM_SAMPLES_TO_ATTACK
    
    print('args_str:', args_str, 'args:', args)
    return args

def parse_model_recipe_from_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('recipe', help='recipe to provide via --recipe')
    parser.add_argument('model', help='model to provide via --model')
    args = parser.parse_args()
    return args.model, args.recipe

def main():
    pytorch_multiprocessing_workaround()
    
    model, recipe = parse_model_recipe_from_args()
    
    try:
        PRECOMPUTED_RESULTS = pickle.load(open(PRECOMPUTED_RESULTS_DICT_NAME, "rb" ))
    except FileNotFoundError:
        PRECOMPUTED_RESULTS = {}
    print(f'Found {len(PRECOMPUTED_RESULTS)} pre-computed results.')
    
    # no need to recompute results
    if (model, recipe) in PRECOMPUTED_RESULTS:
        print(f'found {(model, recipe)} already precomputed - skipping')
        exit()
    
    # also don't do in parallel
    file_lock_dir = os.path.expanduser('~/.cache/textattack-streamlit')
    file_lock_path = os.path.join(file_lock_dir, hashlib.sha224(('precompute' + model + recipe).encode()).hexdigest())
    
    if os.path.exists(file_lock_path):
        print(f'attacking model {model} and recipe {recipe} / found file_lock at path {file_lock_path}; skipping')
        exit()
    else:
        print(f'attacking model {model} and recipe {recipe} / creating file_lock at path {file_lock_path}')
        basedir = os.path.dirname(file_lock_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        open(file_lock_path, 'a').close()
    
    print(f'** running attack / model: {model} / recipe {recipe} **')
    args_str = f'attack --model {model} --recipe {recipe}'
    args = parse_ta_args(args_str)
    if torch.cuda.device_count() > 1:
        attack_results = textattack.commands.attack.run_attack_parallel(args)
    else:
        attack_results = textattack.commands.attack.run_attack_single_threaded(args)
    del args
    PRECOMPUTED_RESULTS[(model, recipe)] = attack_results
    if not attack_results:
        raise RuntimeError(f'missing attack results even though args.num_examples = {args.num_examples}')
    # dump results to attack file
    pickle.dump(PRECOMPUTED_RESULTS, open(PRECOMPUTED_RESULTS_DICT_NAME, 'wb'))
    
    # delete file lock
    print('removing lock at path',file_lock_path)
    os.remove(file_lock_path)

def pytorch_multiprocessing_workaround():
    # This is a fix for a known bug
    try:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

if __name__ == '__main__': main()