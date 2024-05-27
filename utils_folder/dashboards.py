
from pathlib import Path
import time
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass
from transformer_lens import utils, HookedTransformer
from transformer_lens.hook_points import HookPoint
import torch
from torch import Tensor
from eindex import eindex
from IPython.display import display, HTML
from typing import Optional, List, Dict, Callable, Tuple, Union, Literal
from dataclasses import dataclass
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from collections import defaultdict
from functools import partial
from rich import print as rprint
from rich.table import Table
import pickle
import os
from IPython.display import display, HTML, clear_output
clear_output()

import os, sys

from transformer_lens import HookedTransformer, utils
import torch
from datasets import load_dataset
from typing import Dict
from tqdm.notebook import tqdm
import plotly.express as px
import json

# Add the directory containing both the script and the module to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
from sae_visualizer.model_fns import AutoEncoderConfig, AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()


import pprint


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


from torch import nn
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version, device_override=None):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47

        cfg = utils.download_file_from_hf("ckkissane/tinystories-1M-SAES", f"{version}_cfg.json")
        if device_override is not None:
            cfg["device"] = device_override

        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("ckkissane/tinystories-1M-SAES", f"{version}.pt", force_is_torch=True))
        return self
    

from sae_visualizer.utils_fns import sample_unique_indices, TopK, k_largest_indices, random_range_indices, reshape, to_str_tokens


@dataclass
class MinimalSequenceData:
    '''
    Class to store data for a given sequence, which will be turned into a JavaScript visulisation.
    Basically just wraps token_ids and the corresponding feature acts for some (prompt, feature_id) pair

    Before hover:
        str_tokens: list of string tokens in the sequence
        feat_acts: sizes of activations on this sequence

    '''
    token_ids: List[str]
    feat_acts: List[float]

    def __len__(self):
        return len(self.token_ids)

    def __str__(self):
        return f"MinimalSequenceData({''.join(self.token_ids)})"

    def _filter(self, float_list: List[List[float]], int_list: List[List[str]]):
        float_list = [[f for f in floats if f != 0] for floats in float_list]
        int_list = [[i for i, f in zip(ints, floats)] for ints, floats in zip(int_list, float_list)]
        return float_list, int_list

class MinimalSequenceDataBatch:
    '''
    Class to store a list of MinimalSequenceData objects at once, by passing in tensors or objects
    with an extra dimension at the start.

    Note, I'll be creating these objects by passing in objects which are either 2D (k seq_len)
    or 3D (k seq_len top5), but which are all lists (of strings/ints/floats).

    '''
    def __init__(self, **kwargs):
        self.seqs = [
            MinimalSequenceData(
                token_ids = kwargs["token_ids"][k],
                feat_acts = kwargs["feat_acts"][k],
            )
            for k in range(len(kwargs["token_ids"]))
        ]

    def __getitem__(self, idx: int) -> MinimalSequenceData:
        return self.seqs[idx]

    def __len__(self) -> int:
        return len(self.seqs)

    def __str__(self) -> str:
        return "\n".join([str(seq) for seq in self.seqs])
    

@torch.inference_mode()
def get_seq_data(
    encoder: AutoEncoder,
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    feature_idx: Union[int, List[int]],
    max_batch_size: Optional[int] = None,

    buffer: Tuple[int, int] = (5, 5),
    n_groups: int = 10,
    first_group_size: int = 20,
    other_groups_size: int = 5,
    verbose: bool = False,

):
    '''
    Gets data that will be used to create the sequences in the HTML visualisation.

    Args:
        feature_idx: int
            The identity of the feature we're looking at (i.e. we slice the weights of the encoder). A list of
            features is accepted (the result will be a list of FeatureData objects).
        max_batch_size: Optional[int]
            Optionally used to chunk the tokens, if it's a large batch

        buffer: Tuple[int, int]
            The number of tokens on either side of the feature, for the right-hand visualisation.

    Returns list of dictionaries that contain SequenceBatchData for each feature (see that class's docstring for more info).
    '''
    model.reset_hooks(including_permanent=True)

    # Make feature_idx a list, for convenience
    if isinstance(feature_idx, int): feature_idx = [feature_idx]
    n_feats = len(feature_idx)

    # Chunk the tokens, for less memory usage
    all_tokens = (tokens,) if max_batch_size is None else tokens.split(max_batch_size)
    all_tokens = [tok.to(device) for tok in all_tokens]

    # Create lists to store data, which we'll eventually concatenate
    all_feat_acts = []

    # Get encoder & decoder directions
    feature_act_dir = encoder.W_enc[:, feature_idx] # (d_mlp, feats)
    feature_bias = encoder.b_enc[feature_idx] # (feats,)
    #assert feature_act_dir.T.shape == feature_out_dir.shape #== (len(feature_idx), encoder.cfg.d_mlp)

    # ! Define hook function to perform feature ablation
    def hook_fn_act_post(act: Float[Tensor, "batch seq act_size"], hook: HookPoint):
        '''
        Encoder has learned x^j \approx b + \sum_i f_i(x^j)d_i where:
            - f_i are the feature activations
            - d_i are the feature output directions

        This hook function stores all the information we'll need later on. It doesn't actually perform feature ablation, because
        if we did this, then we'd have to run a different fwd pass for every feature, which is super wasteful! But later, we'll
        calculate the effect of feature ablation,  i.e. x^j <- x^j - f_i(x^j)d_i for i = feature_idx, only on the tokens we care
        about (the ones which will appear in the visualisation).
        '''
        #if encoder.cfg["concat_heads"]:
        act = einops.rearrange(
            act, "batch seq n_heads d_head -> batch seq (n_heads d_head)",
        )
        x_cent = act - encoder.b_dec
        feat_acts_pre = einops.einsum(x_cent, feature_act_dir, "batch seq act_size, act_size feats -> batch seq feats")
        feat_acts = F.relu(feat_acts_pre + feature_bias)
        all_feat_acts.append(feat_acts)

    # ! Run the forward passes (triggering the hooks), concat all results

    # Run the model without hook (to store all the information we need, not to actually return anything)
    for _tokens in all_tokens:
        model.run_with_hooks(_tokens, return_type=None, fwd_hooks=[
            (encoder.cfg.hook_point, hook_fn_act_post),
        ])

    # Stack the results, and check shapes
    feat_acts = torch.concatenate(all_feat_acts) # [batch seq feats]
    assert feat_acts[:, :-1].shape == tokens[:, :-1].shape + (len(feature_idx),)

    # ! Calculate all data for the right-hand visualisations, i.e. the sequences
    # TODO - parallelize this (it could probably be sped up by batching indices & doing all sequences at once, although those would be large tensors)
    # We do this in 2 steps:
    #   (1) get the indices per group, from the feature activations, for each of the 12 groups (top, bottom, 10 quantiles)
    #   (2) get a batch of SequenceData objects per group. This usually involves using eindex (i.e. indexing into the `tensors`
    #       tensor with the group indices), and it also requires us to calculate the effect of ablations (using feature activations
    #       and the clean residual stream values).

    sequence_data_list = []

    iterator = range(n_feats) if not(verbose) else tqdm(range(n_feats), desc="Getting sequence data", leave=False)

    for feat in iterator:

        _feat_acts = feat_acts[..., feat] # [batch seq]

        # (1)
        indices_dict = {
            f"TOP ACTIVATIONS<br>MAX = {_feat_acts.max():.3f}": k_largest_indices(_feat_acts, k=first_group_size, largest=True),
            f"BOTTOM ACTIVATIONS<br>MIN = {_feat_acts.min():.3f}": k_largest_indices(_feat_acts, k=first_group_size, largest=False),
        }

        quantiles = torch.linspace(0, _feat_acts.max(), n_groups+1)
        for i in range(n_groups-1, -1, -1):
            lower, upper = quantiles[i:i+2]
            pct = ((_feat_acts >= lower) & (_feat_acts <= upper)).float().mean()
            indices = random_range_indices(_feat_acts, (lower, upper), k=other_groups_size)
            indices_dict[f"INTERVAL {lower:.3f} - {upper:.3f}<br>CONTAINS {pct:.3%}"] = indices

        # Concat all the indices together (in the next steps we do all groups at once)
        indices_full = torch.concat(list(indices_dict.values()))

        # (2)
        # ! We further split (2) up into 3 sections:
        #   (A) calculate the indices we'll use for this group (we need to get a buffer on either side of the target token for each seq),
        #       i.e. indices[..., 0] = shape (g, buf) contains the batch indices of the sequences, and indices[..., 1] = contains seq indices
        #   (B) index into all our tensors to get the relevant data (this includes calculating the effect of ablation)
        #   (C) construct the SequenceData objects, in the form of a SequenceDataBatch object

        # (A)
        # For each token index [batch, seq], we actually want [[batch, seq-buffer[0]], ..., [batch, seq], ..., [batch, seq+buffer[1]]]
        # We get one extra dimension at the start, because we need to see the effect on loss of the first token
        buffer_tensor = torch.arange(-buffer[0] - 1, buffer[1] + 1, device=indices_full.device)
        indices_full = einops.repeat(indices_full, "g two -> g buf two", buf=buffer[0] + buffer[1] + 2)
        indices_full = torch.stack([indices_full[..., 0], indices_full[..., 1] + buffer_tensor], dim=-1).cpu()

        # (B)
        # Template for indexing is new_tensor[k, seq] = tensor[indices_full[k, seq, 1], indices_full[k, seq, 2]], sometimes there's an extra dim at the end
        tokens_group = eindex(tokens, indices_full[:, 1:], "[g buf 0] [g buf 1]")
        feat_acts_group = eindex(_feat_acts, indices_full, "[g buf 0] [g buf 1]")

        # (C)
        # Now that we've indexed everything, construct the batch of SequenceData objects
        sequence_data = {}
        g_total = 0
        for group_name, indices in indices_dict.items():
            lower, upper = g_total, g_total + len(indices)
            sequence_data[group_name] = MinimalSequenceDataBatch(
                token_ids=tokens_group[lower: upper].tolist(),
                feat_acts=feat_acts_group[lower: upper, 1:].tolist(),
            )
            g_total += len(indices)

        # Add this feature's sequence data to the list
        sequence_data_list.append(sequence_data)

    return sequence_data_list


base_dir = Path(__file__).resolve().parent
HTML_DIR = base_dir / "sae_visualizer" / "html"

#HTML_DIR = Path("sae_visualizer/html")
HTML_HOVERTEXT_SCRIPT = (HTML_DIR / "hovertext_script.html").read_text()
#HTML_TOKEN = (HTML_DIR / "token_template.html").read_text()
HTML_TOKEN = """<span class="hover-text">
    <span class="token" style="background-color: bg_color; border-bottom: 4px solid underline_color; font-weight: font_weight">this_token</span>
    <div class="tooltip">
        <div class="table-container">
            <table>
                <tr><td class="right-aligned">Token</td><td class="left-aligned"><code>this_token</code></td></tr>
                <tr><td class="right-aligned">Feature activation</td><td class="left-aligned">feat_activation</td></tr>
            </table>
            <!-- No effect! -->
          </div>
    </div>
</span>"""

CSS_DIR = base_dir / "sae_visualizer" / "css"

CSS = "\n".join([
    (CSS_DIR / "general.css").read_text(),
    (CSS_DIR / "sequences.css").read_text(),
    (CSS_DIR / "tables.css").read_text(),
])

@dataclass
class LogitData:
  top_negative_logits: List[Tuple[str, float]]
  top_positive_logits: List[Tuple[str, float]]

@dataclass
class DecoderWeightsDistribution:
  n_heads: int
  allocation_by_head: List[float]

from matplotlib import colors
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import re

BG_COLOR_MAP = colors.LinearSegmentedColormap.from_list("bg_color_map", ["white", "darkorange"])

def generate_minimal_tok_html(
    vocab_dict: dict,
    this_token: str,
    bg_color: str,
    is_bold: bool = False,
    feat_act: float = 0.0,
):
    '''
    Creates a single sequence visualisation, by reading from the `token_template.html` file.

    Currently, a bunch of things are randomly chosen rather than actually calculated (we're going for
    proof of concept here).
    '''
    html_output = (
        HTML_TOKEN
        .replace("this_token", to_str_tokens(vocab_dict, this_token))
        .replace("feat_activation", f"{feat_act:+.3f}")
        .replace("font_weight", "bold" if is_bold else "normal")
        .replace("bg_color", bg_color)
    )
    return html_output

def generate_minimial_seq_html(
    vocab_dict: dict,
    token_ids: List[str],
    feat_acts: List[float],
    bold_idx: Optional[int] = None,
):
    assert len(token_ids) == len(feat_acts), "All input lists must be of the same length."

    # ! Clip values in [0, 1] range (temporary)
    bg_values = np.clip(feat_acts, 0, 1)


    # Define the HTML object, which we'll iteratively add to
    html_output = '<div class="seq">'  # + repeat_obj

    for i in range(len(token_ids)):

        # Get background color, which is {0: transparent, +1: darkorange}
        bg_val = bg_values[i]
        bg_color = colors.rgb2hex(BG_COLOR_MAP(bg_val))

        html_output += generate_minimal_tok_html(
            vocab_dict = vocab_dict,
            this_token = token_ids[i],
            bg_color = bg_color,
            is_bold = (bold_idx is not None) and (bold_idx == i),
            feat_act = feat_acts[i],
        )

    html_output += '</div>'
    return html_output

# should be able to pass feature_data[0] to this
def get_minimal_sequences_html(sequence_data, vocab_dict, bold_idx,
                               logits_data: LogitData,
                               decoder_weights_distribution: DecoderWeightsDistribution, model) -> str:

    sequences_html_dict = {}


    for group_name, sequences in sequence_data.items():

        full_html = f'<h4>{group_name}</h4>' # style="padding-left:25px;"

        for seq in sequences:
            html_output = generate_minimial_seq_html(
                vocab_dict,
                token_ids = seq.token_ids,
                feat_acts = seq.feat_acts,
                bold_idx = bold_idx, # e.g. the 6th item, with index 5, if buffer=(5, 5)
            )
            full_html += html_output

        sequences_html_dict[group_name] = full_html

    # Now, wrap all the values of this dictionary into grid-items: (top, groups of 3 for middle, bottom)
    html_top, html_bottom, *html_sampled = sequences_html_dict.values()
    sequences_html = ""
    sequences_html += f"<div class='grid-item'>{html_top}</div>"

    # Add decoder weights distribution
    sequences_html += f"<div class='grid-item'><h4>Decoder Weights Distribution</h4>"
    for i in range(model.cfg.n_heads):
      sequences_html += f"<div style='display: inline-block; border-bottom: 1px solid #ccc'>Head {i}: {decoder_weights_distribution.allocation_by_head[i]:0.2f}</div><br />"
    sequences_html += "</div>"

    # First add top/bottom logits card
    opacity = lambda logit, logits: (logit - \
        (m := min(logit for _, logit in logits))) / \
        (max(logit for _, logit in logits) - m) / 4 + 0.5
    positive_opacity = partial(opacity, logits=logits_data.top_positive_logits)
    negative_opacity = partial(opacity, logits=logits_data.top_negative_logits)
    positive_logits_html = "".join(
      f"<div style='display: inline-block; border-bottom: 1px solid #ccc; width: 100%'><span style='background-color: rgba(171, 171, 255, {positive_opacity(logit)}); float: left;'>{token}</span><span style='float: right'>{logit:0.2f}</span></div><br />"
      for token, logit in logits_data.top_positive_logits
    )
    sequences_html += f"<div class='grid-item'><h4>Positive logits</h4>{positive_logits_html}</div>"
    negative_logits_html = "".join(
      f"<div style='display: inline-block; border-bottom: 1px solid #ccc; width: 100%'><span style='background-color: rgba(255, 149, 149, {negative_opacity(logit)}); float: left;'>{token}</span><span style='float: right'>{logit:0.2f}</span></div><br />"
      for token, logit in logits_data.top_negative_logits
    )
    sequences_html += f"<div class='grid-item'><h4>Negative logits</h4>{negative_logits_html}</div>"

    while len(html_sampled) > 0:
        L = min(3, len(html_sampled))
        html_next, html_sampled = html_sampled[:L], html_sampled[L:]
        sequences_html += "<div class='grid-item'>" + "".join(html_next) + "</div>"
    sequences_html += f"<div class='grid-item'>{html_bottom}</div>"

    return sequences_html + HTML_HOVERTEXT_SCRIPT

def style_minimal_sequences_html(sequences_html):
    return f"""
<style>
{CSS}
</style>

<div class='grid-container'>

    {sequences_html}

</div>
"""

def get_vocab_dict(model):
    vocab_dict = model.tokenizer.vocab
    vocab_dict = {v: k.replace("Ġ", " ").replace("\n", "\\n") for k, v in vocab_dict.items()}
    return vocab_dict




def get_logit_data(encoder: AutoEncoder, model: nn.Module,
                   layer_ix: int, feature_idx: Union[int, List[int]],
                   top_k: int=20) -> List[LogitData]:
  if not isinstance(feature_idx, list):
    feature_idx = [feature_idx]

  logit_data = []
  flattened_WO = einops.rearrange(
    model.blocks[layer_ix].attn.W_O,
    "n_head d_head d_resid -> (n_head d_head) d_resid"
  )

  for feature in feature_idx:
    logits = encoder.W_dec[feature, :] @ flattened_WO @ model.W_U
    positive_logits = logits.topk(k=top_k, largest=True, sorted=True)
    negative_logits = logits.topk(k=top_k, largest=False, sorted=True)
    top_positive_logits = []
    top_negative_logits = []
    for token_ix, logit in zip(positive_logits.indices, positive_logits.values):
      top_positive_logits.append((model.tokenizer.decode(token_ix), float(logit)))
    for token_ix, logit in zip(negative_logits.indices, negative_logits.values):
      top_negative_logits.append((model.tokenizer.decode(token_ix), float(logit)))

    logit_data.append(LogitData(top_negative_logits, top_positive_logits))

  return logit_data



def get_decoder_weights_distribution(encoder: AutoEncoder, model: nn.Module,
    layer_ix: int, feature_idx: Union[int, List[int]]) -> List[DecoderWeightsDistribution]:
  if not isinstance(feature_idx, list):
    feature_idx = [feature_idx]

  distribs = []
  for feature in feature_idx:
    att_blocks = einops.rearrange(
        encoder.W_dec[feature, :], "(n_head d_head) -> n_head d_head", n_head=model.cfg.n_heads
    )
    decoder_weights_distribution = att_blocks.norm(dim=1) / att_blocks.norm(dim=1).sum()
    distribs.append(DecoderWeightsDistribution(model.cfg.n_heads, [float(x) for x in decoder_weights_distribution]))

  return distribs


def get_model_card(feature_idx: int, tokens: torch.Tensor, encoder, model, layer_ix, max_batch_size=32):
  feature_data = get_seq_data(
      encoder ,
      model ,
      feature_idx = feature_idx,
      max_batch_size = max_batch_size,
      buffer = (5, 5),
      n_groups = 10,
      first_group_size = 20,
      other_groups_size = 5,
      verbose = True,
      tokens = tokens
  )

  logits_data: List[LogitData] = get_logit_data(
      encoder ,
      model ,
      layer_ix = layer_ix,
      feature_idx = feature_idx,
  )

  decoder_weights_distribution: List[DecoderWeightsDistribution] = \
  get_decoder_weights_distribution(
      encoder ,
      model ,
      layer_ix = layer_ix,
      feature_idx = feature_idx
  )

  vocab_dict = get_vocab_dict(model)
  sub_index = 0
  sequences_html = get_minimal_sequences_html(feature_data[sub_index], vocab_dict,
                                              bold_idx=6, logits_data=logits_data[sub_index],
                                              decoder_weights_distribution=decoder_weights_distribution[sub_index],
                                              model=model)
  html_string = style_minimal_sequences_html(sequences_html)
  display(HTML(html_string))
