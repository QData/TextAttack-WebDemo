from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerDeepLiftShap, InternalInfluence, LayerGradientXActivation
import torch
from copy import deepcopy
from captum.attr import visualization as viz

import textattack
import transformers

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

def captum_form(encoded, device):
    input_dict = {k: [_dict[k] for _dict in encoded] for k in encoded[0]}
    batch_encoded = { k: torch.tensor(v).to(device) for k, v in input_dict.items()}
    return batch_encoded

def formatDisplay(datarecords):
    rows = []
    for datarecord in datarecords:
        rows.append(
            viz.format_word_importances(
                datarecord.raw_input, datarecord.word_attributions
            )
        )

    return rows

original_model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
original_tokenizer = textattack.models.tokenizers.AutoTokenizer("textattack/bert-base-uncased-ag-news")
model = textattack.models.wrappers.HuggingFaceModelWrapper(original_model,original_tokenizer)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
clone = deepcopy(model)
clone.model.to(device)

def calculate(input_ids, token_type_ids, attention_mask):
    return clone.model(input_ids,token_type_ids,attention_mask)[0]

attack = textattack.commands.attack.attack_args_helpers.parse_attack_from_args(Args("bert-base-uncased-ag-news", "alzantot"))
attacked_text = textattack.shared.attacked_text.AttackedText("Hello World")
attack.goal_function.init_attack_example(attacked_text, 1)
goal_func_result, _ = attack.goal_function.get_result(attacked_text)

result = next(attack.attack_dataset([("Hello World", goal_func_result.output)]))

orig = result.original_text()
pert = result.perturbed_text()

encoded = model.tokenizer.batch_encode([orig])
batch_encoded = captum_form(encoded, device)
x = calculate(**batch_encoded)

pert_encoded = model.tokenizer.batch_encode([pert])
pert_batch_encoded = captum_form(pert_encoded, device)
x_pert = calculate(**pert_batch_encoded) 

lig = LayerIntegratedGradients(calculate, clone.model.bert.embeddings)
attributions,delta = lig.attribute(inputs=batch_encoded['input_ids'],
                        additional_forward_args=(batch_encoded['token_type_ids'], batch_encoded['attention_mask']),
                        n_steps = 10,
                        target = torch.argmax(calculate(**batch_encoded)).item(),
                        return_convergence_delta=True
                    )

attributions_pert,delta_pert = lig.attribute(inputs=pert_batch_encoded['input_ids'],
                        additional_forward_args=(pert_batch_encoded['token_type_ids'], pert_batch_encoded['attention_mask']),
                        n_steps = 10,
                        target = torch.argmax(calculate(**pert_batch_encoded)).item(),
                        return_convergence_delta=True
                    )

orig = original_tokenizer.tokenizer.tokenize(orig)
pert = original_tokenizer.tokenizer.tokenize(pert)

atts = attributions.sum(dim=-1).squeeze(0)
atts = atts / torch.norm(atts)
        
atts_pert = attributions_pert.sum(dim=-1).squeeze(0)
atts_pert = atts_pert / torch.norm(atts)                

all_tokens = original_tokenizer.tokenizer.convert_ids_to_tokens(batch_encoded['input_ids'][0])
all_tokens_pert = original_tokenizer.tokenizer.convert_ids_to_tokens(pert_batch_encoded['input_ids'][0])

v = viz.VisualizationDataRecord(
                atts[:45].detach().cpu(),
                torch.max(x).item(),
                torch.argmax(x,dim=1).item(),
                goal_func_result.output,
                2,
                atts.sum().detach(), 
                all_tokens[:45],
                delta)

v_pert = viz.VisualizationDataRecord(
                atts_pert[:45].detach().cpu(),
                torch.max(x_pert).item(),
                torch.argmax(x_pert,dim=1).item(),
                goal_func_result.output,
                2,
                atts_pert.sum().detach(), 
                all_tokens_pert[:45],
                delta_pert)

viz_list = [v, v_pert]

print(formatDisplay(viz_list))
