import os
import sys
import tqdm

from config import MODELS, ATTACK_RECIPES, HIDDEN_ATTACK_RECIPES

def main():
    FLAGS = ' '.join(sys.argv[1:])
    print('running with flags:', FLAGS)
    for recipe in tqdm.tqdm(ATTACK_RECIPES, desc='Attacking...'):
        # no need to precompute hidden recipes (alzantot is *so* slow!)
        if recipe in HIDDEN_ATTACK_RECIPES:
            print(f'skipping recipe *{recipe}*')
            continue
        print(f'**** Recipe: {recipe} ****')
        for model in tqdm.tqdm(MODELS):
            command = f'{FLAGS} python precompute_one.py {recipe} {model}'
            print('>', command)
            os.system(command)
            print()

if __name__ == '__main__': main()