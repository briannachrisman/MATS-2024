import os
import sys
import torch
sys.path.append("..")
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

# -----------------------------------------------------------------------------
# default config values
model_name = "gpt2-small" 
dataset_path = "Skylion007/openwebtext"

total_training_steps = 500_000 
batch_size = 4096 
new_cached_activations_path = (
    f"./cached_activations/{model_name}/{dataset_path}/{total_training_steps}"
)

hook_point_layer= 6
hook_point=f"blocks.{hook_point_layer}.attn.hook_z"
hook_point_head_index = 6
l1_coefficients = [2, 5, 10]
d_in= 64
expansion_factor = 32

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

total_training_tokens = total_training_steps * batch_size

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training steps.
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warmup_steps = total_training_steps // 20  # 5% of training steps.
print(f"l1_warmup_steps: {l1_warmup_steps}")
log_to_wandb = True

for l1_coefficient in l1_coefficients:

    cfg = LanguageModelSAERunnerConfig(
        # Pick a tiny model to make this easier.
        model_name=model_name, 
        hook_point=hook_point, 
        hook_point_layer=hook_point_layer, 
        hook_point_head_index=hook_point_head_index,
        d_in=d_in,
        dataset_path=dataset_path,
        streaming=False,
        context_size=1024,
        is_dataset_tokenized=False,
        prepend_bos=True,
        # How big do we want our SAE to be?
        expansion_factor=expansion_factor,
        # Dataset / Activation Store
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        use_cached_activations=False,
        #cached_activations_path="./gelu-2l",
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=batch_size,
        # Loss Function
        ## Reconstruction Coefficient.
        mse_loss_normalization=None,  # MSE Loss Normalization is not mentioned (so we use stanrd MSE Loss). But not we take an average over the batch.
        ## Anthropic does not mention using an Lp norm other than L1.
        l1_coefficient=l1_coefficient,
        lp_norm=1.0,
        # Instead, they multiply the L1 loss contribution
        # from each feature of the activations by the decoder norm of the corresponding feature.
        scale_sparsity_penalty_by_decoder_norm=True,
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps. # TODO: understand why it's constant
        l1_warm_up_steps=l1_warmup_steps,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        use_ghost_grads=False,
        # Initialization / Architecture
        apply_b_dec_to_input=False,
        # encoder bias zero's. (I'm not sure what it is by default now)
        # decoder bias zero's.
        b_dec_init_method="zeros",
        normalize_sae_decoder=False,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        # Optimizer
        lr=5e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        normalize_activations=True, # TODO: why is the scaling factor 1.0 even when this is True?
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # WANDB
        log_to_wandb=log_to_wandb,  # always use wandb unless you are just testing code.
        wandb_project=f"{model_name}-attn-{hook_point_layer}-sae",
        wandb_log_frequency=50,
        eval_every_n_wandb_logs=10,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints",
        dtype=torch.float32,
    )

    print(f"Total Training Tokens: {total_training_tokens}")

    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder_dictionary = language_model_sae_runner(cfg)

    print("=" * 50)