import einops
import os
from transformers import AutoModel, AutoTokenizer
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes, convert_connor_rob_sae_to_our_saelens_format, download_sae_from_hf
import numpy as np
import torch as t
import plotly_express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.notebook import tqdm
from scipy import stats
from torch.utils.data import Dataset


from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.utils import (
    load_dataset,
    tokenize_and_concatenate,
    download_file_from_hf,
)

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from transformer_lens import HookedSAE, HookedSAEConfig
from transformer_lens.utils import download_file_from_hf

# Token dataset loader.
class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return (token, idx)



# Function to take a dictionary of {a:[b]} and return {b:[a]}.
def ReverseDict(original_dict):
    new_dict = dict()
    for k in original_dict:
        for v in original_dict[k]:
            if v.item() not in new_dict:
                new_dict[v.item()] = []
            new_dict[v.item()] = new_dict[v.item()] + [k]
    return {k:set(new_dict[k]) for k in new_dict}



# Function to take an attention SAE config and return a hooked SAE.
def AttnToHookedCfg(attn_sae_cfg):
    new_cfg = {
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "hook_name": attn_sae_cfg["act_name"],
    }
    return HookedSAEConfig.from_dict(new_cfg)


def GetHeadContributions(concat_vector):
    splits = torch.split(concat_vector, model.cfg.d_head, dim=1)

    # Turn list of tensors into a single tenso
    splits = torch.stack(splits, dim=0)
    contributions = (splits**2).sum(dim=2)
    contributions = contributions/contributions.sum(dim=0, keepdim=True)
    return contributions