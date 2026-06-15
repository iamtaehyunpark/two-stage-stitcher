from dataclasses import dataclass


@dataclass
class StitcherConfig:
    # Models
    source_model: str = "Qwen/Qwen2.5-7B-Instruct"
    target_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # Dimensions
    src_dim: int = 3584   # Qwen2.5-7B hidden size
    tgt_dim: int = 8192   # DeepSeek-R1-Distill-Llama-70B hidden size

    # Which target-model layer to extract from / inject into (0-indexed, out of 80)
    target_layer: int = 30

    # Data collection
    chunk_size: int = 1024           # tokens per progressive chunk
    max_chunks_per_doc: int = 16     # up from 8 — more pairs per doc
    data_dir: str = "data/hidden_states"
    output_dir: str = "checkpoints"

    # SVD alignment (Stage 1)
    svd_checkpoint: str = "checkpoints/w_optimal.pt"

    # MLP (Stage 2) — keep original architecture to stay compatible with existing checkpoints
    mlp_hidden_dim: int = 4096
    mlp_num_layers: int = 3
    mlp_dropout: float = 0.0

    # Training
    batch_size: int = 256            # up from 128 — more InfoNCE negatives
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    num_epochs: int = 50             # up from 10
    warmup_steps: int = 2000         # up from 500
    grad_clip: float = 1.0
    lambda_mse: float = 0.5          # up from 0.1 — stronger MSE anchor
    infonce_temperature: float = 0.05  # down from 0.07 — sharper contrastive boundary

    # Training — early stopping
    early_stop_patience: int = 7     # stop if val_loss doesn't improve for this many epochs
    early_stop_min_delta: float = 1e-4  # minimum improvement to count as progress

    # Validation
    top_k_retrieval: int = 1

    # Hardware
    # These are *logical* indices within CUDA_VISIBLE_DEVICES, not physical GPU IDs.
    # Set CUDA_VISIBLE_DEVICES in the environment to select which physical GPUs to use.
    # Layout: cuda:3 = Qwen + MLP, cuda:0-2 = DeepSeek shards
    source_device: str = "cuda:3"
    llama_devices: tuple = (0, 1, 2)
    dtype: str = "bfloat16"
