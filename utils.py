import torch
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
from pathlib import Path 
from transformer_lens import utils
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
# from utils import CR_AutoEncoder
from sae_lens.toolkit.pretrained_saes import convert_connor_rob_sae_to_our_saelens_format
from dataclasses import asdict
from torch import nn
from tqdm import tqdm
# SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
torch.set_grad_enabled(False)
# I don't fully understand this but it seems important to avoid some warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "gelu-2l"
hook_point_layer=1
hook_point=f"blocks.{hook_point_layer}.attn.hook_z"
d_in= 64
expansion_factor = 32
sae_name = f"{model_name}_{hook_point}_{d_in * expansion_factor}_"

# Their SAE code is taken from the Colab Notebook
# https://colab.research.google.com/drive/10zBOdozYR2Aq2yV9xKs-csBH2olaFnsq?usp=sharing#scrollTo=feJOqPeoPjvX
# The SAE name is in "Loading models and data section"
class CR_AutoEncoder(nn.Module): # Connor and Rob's Autoencoder class
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

        # pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("ckkissane/tinystories-1M-SAES", f"{version}.pt", force_is_torch=True))
        return self


def standardize(data):
    """ Standardize the data to have mean 0 and standard deviation 1 along the first axis """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True, unbiased=False)
    return (data - mean) / std

def compute_corr_matrix(latents1, latents2):

    assert latents1.shape[0] == latents2.shape[0], "number of data points need to be the same"

    # Standardize latents1 and latents2
    latents1_standard = standardize(latents1)
    latents2_standard = standardize(latents2)

    # TODO: decide whether to divide by latents1.shape[0] or latents1.shape[0] - 1
    # Compute the dot product of the transposes of the standardized matrices
    corr_matrix = torch.mm(latents1_standard.t(), latents2_standard) / (latents1.shape[0]-1)

    #print(f"correlation_matrix has shape: {corr_matrix.shape}")

    return corr_matrix

#------------------


# ckpt_subfolders = { 
#     0: "rovi1lwe", #3.785
#     1: "p7113j0v", #3.807
#     2: "rjc53kjg", #3.768
#     3: "hibm6x1l", #3.738
#     4: "4xima76s", #3.746
#     5: "jq26bfpa", #3.729
#     6: "b8e2a9w5", #3.75
#     7: "smfws6mc" # 3.748
# }
# ckpt_id = "983044096"

def get_ckpt_dir(hook_point_head_index):
    ckpt_subfolders = {0: "b6pycfom",
    1: "kfe4unl4",
    2: "8g2358dj",
    3: "wyg4xt9k",
    4: "qybu26ew",
    5: "i6i03t2h",
    6: "0vbh9ov0",
    7: "bfus7egg"}
    ckpt_id = "final_1024000000"
    ckpt_dir = os.path.join("checkpoints",
                        ckpt_subfolders[hook_point_head_index],
                        ckpt_id,
                        sae_name)
    return ckpt_dir

# concatenate our SAEs into an SAE that decomposes concatenated activations


def load_our_sae(hook_point_head_index):
    print(f"Loading SAE for head # {hook_point_head_index}")
    ckpt_dir = get_ckpt_dir(hook_point_head_index)
    model, sae, activations_loader = LMSparseAutoencoderSessionloader.load_pretrained_sae(path=ckpt_dir,
                                                                                          device=device)
    activations_loader.device = "cpu"
    sae = sae.autoencoders[sae_name]
    sae.eval()
    return model, sae, activations_loader

def load_our_saes():
    # load SAEs for each attention head
    n_heads = 8
    saes = {}
    for hook_point_head_index in range(n_heads):
        model, sae, activations_loader = load_our_sae(hook_point_head_index)
        saes[hook_point_head_index] = sae
    return model, saes, activations_loader

def load_concatenated_sae():
    model, saes, activations_loader = load_our_saes()
    print(f"Concatenating all SAEs from individual attention heads")
    cat_sae_cfg = asdict(saes[0].cfg)
    cat_sae_cfg["hook_point_head_index"] = None
    cat_sae_cfg["d_in"] = 64 * 8
    cat_sae_cfg["d_sae"] = 2048 * 8
    cat_sae_cfg = LanguageModelSAERunnerConfig(**cat_sae_cfg)
    
    cat_sae = SparseAutoencoder(cat_sae_cfg)
    
    # weights are block-diagonal and biases are just concatenations
    cat_sae.W_dec = torch.nn.Parameter(torch.block_diag(*[sae.W_dec for head, sae in sorted(saes.items())]))
    cat_sae.W_enc = torch.nn.Parameter(torch.block_diag(*[sae.W_enc for head, sae in sorted(saes.items())]))
    cat_sae.b_dec = torch.nn.Parameter(torch.cat(tuple(sae.b_dec.data for head, sae in sorted(saes.items()))))
    cat_sae.b_enc = torch.nn.Parameter(torch.cat(tuple(sae.b_enc.data for head, sae in sorted(saes.items()))))
    return model, cat_sae, activations_loader

def import_connor_sae():
    # import Connor and Rob's SAE
    auto_encoder_run = "concat-z-gelu-21-l1-lr-sweep-3/gelu-2l_L1_Hcat_z_lr1.00e-03_l12.00e+00_ds16384_bs4096_dc1.00e-07_rie50000_nr4_v78"
    connor_rob_sae = CR_AutoEncoder.load_from_hf(auto_encoder_run)
    # New sae-lens state dict requires scaling factor which CR'SAE did not have
    connor_rob_sae_state_dict = connor_rob_sae.state_dict()
    connor_rob_sae_state_dict["scaling_factor"] = torch.ones(connor_rob_sae.cfg["dict_size"],)
    
    connor_rob_sae = convert_connor_rob_sae_to_our_saelens_format(
        state_dict=connor_rob_sae_state_dict,
        config=connor_rob_sae.cfg,
    ).to(device=device)
    return connor_rob_sae

def get_tokens(
    activation_store: ActivationsStore,
    n_batches_to_sample_from: int = 2**10,
    n_prompts_to_select: int = 4096 * 6,
    ):
    all_tokens_list = []
    pbar = tqdm(range(n_batches_to_sample_from))
    for _ in pbar:
        batch_tokens = activation_store.get_batch_tokens(batch_size=64) # (batch_size (default 16), 1024)
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ] # randomly shuffle batches (16, 1024)
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0) # convert into a tensor (16 * n_batches_to_sample_from, 1024)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens[:n_prompts_to_select] # (n_prompts_to_select, 1024)