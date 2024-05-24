import torch
from pathlib import Path 
from transformer_lens import utils
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

from torch import nn
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")

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