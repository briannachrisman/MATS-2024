import tqdm
import einops
import torch as t
from torch.utils.data import DataLoader
import utils
import numpy as np


def ActivatingExamples(sae, model, tokens, hook_point, batch_size=64):
    '''
    Get the examples from a token dataset that activate an SAE's features. 
    Args:
    sae: The SAE whose features we want to find.
    model: The model whose activations we want to use.
    tokens: The token dataset.
    hook_point: The layer at which we want to compute activations.
    batch_size: The batch size for the DataLoader.
    Returns:
    A dictionary of the form {feature_id:[text_ids]}.
    '''
    model.eval()
    text_to_features_dict = dict()
    text_to_features_tensor = t.zeros(len(tokens), sae.W_dec.shape[0])
    for tokens_batch, tokens_idx in DataLoader(tokens, batch_size=batch_size):
        # Memory management.
        with t.no_grad():
            t.cuda.empty_cache()
            
            # Get cache & activations at hook point.
            logits, cache = model.run_with_cache(tokens_batch, names_filter=[hook_point])
            activations = cache[hook_point].to('cpu')

            del cache, logits # Delete from GPU to save memory.
            
            # Compute whether or not a feature was activated by a given set of activations.
            try:
                features = (sae.forward(activations).sum(dim=1)>0).float()
            except:
                features = (sae.forward(activations).feature_acts.sum(dim=1)>0).float()

            text_to_features_tensor[tokens_idx,:] = features

            # Build dictionaries containing {text_id:[features_activated]}
            for within_batch_idx, token_id in enumerate(tokens_idx):
                text_to_features_dict[token_id.item()] = set(t.where(features[within_batch_idx,:]>0)[0])
    
    return utils.ReverseDict(text_to_features_dict), text_to_features_tensor




def MutualInformation(activating_examples_tensor_1, activating_examples_tensor_2):
    '''
    Using dictionarires of the form {feature:[ids that activated it]]}, compute the mutual information between the two sets of features.
    Args:
    activating_examples_1: The first dictionary.
    activating_examples_2: The second dictionary.
    Returns:
    The mutual information between the two sets of features.
    '''
   
    mutual_info = dict()
    
    p1 = activating_examples_tensor_1.sum(dim=0, keepdim=True)
    p2 = activating_examples_tensor_2.sum(dim=0, keepdim=True)
    
    p12 = einops.einsum(activating_examples_tensor_1, activating_examples_tensor_2, 'text feature1 , text feature2-> feature1 feature2')
    
    MI = p12/(p1*p2)


    MI = MI*MI.log()

    idx = t.tril_indices(MI.shape[0], MI.shape[1], offset=-1)

    # Get the elements of cosine at idx
    MI = MI[idx[0,:], idx[1,:]]
    return MI, idx

    
    for feature1, ids1 in tqdm.tqdm(activating_examples_1.items()):

        for feature2, ids2 in activating_examples_2.items():
            p = len(ids1.intersection(ids2)) / (len(ids1)*len(ids2))
            mutual_info[(feature1, feature2)] = p * np.log(p)

    return mutual_info



def TokenActivations(sae, model, tokens, hook_point):
    '''
    Get the examples from a token dataset that activate an SAE's features. 
    Args:
    sae: The SAE whose features we want to find.
    model: The model whose activations we want to use.
    tokens: The token dataset.
    hook_point: The layer at which we want to compute activations.
    batch_size: The batch size for the DataLoader.
    Returns:
    A dictionary of the form {feature_id:[text_ids]}.
    '''
    model.eval()
    text_to_features_dict = dict()
    # Memory management.
    with t.no_grad():
        t.cuda.empty_cache()
                
        # Get cache & activations at hook point.
        logits, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        activations = cache[hook_point]

        del cache, logits # Delete from GPU to save memory.
                
        # Compute how much a feature was activated by each token.
        try:
            feature_vals = sae.forward(activations).float()
        except:
            feature_vals = sae.forward(activations).feature_acts.float()

        return {model.tokenizer.decode(i.item()):j.item() for i,j in zip(tokens.flatten(), feature_vals.flatten())}