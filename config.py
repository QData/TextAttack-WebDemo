import textattack

NUM_SAMPLES_TO_ATTACK = 100
MODELS = textattack.commands.attack.attack_args.HUGGINGFACE_DATASET_BY_MODEL
ATTACK_RECIPES =  textattack.commands.attack.attack_args.ATTACK_RECIPE_NAMES
HIDDEN_ATTACK_RECIPES = ['alzantot', 'seq2sick', 'hotflip']

PRECOMPUTED_RESULTS_DICT_NAME = 'results.p'
HISTORY = 'timeline_history'
