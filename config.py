from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StitcherConfig:
    # Models
    source_model: str = "Qwen/Qwen2.5-7B-Instruct"
    target_model: str = "meta-llama/Llama-3.1-70B-Instruct"

    # Dimensions
    src_dim: int = 3584   # Qwen2.5-7B hidden size
    tgt_dim: int = 8192   # Llama-70B hidden size

    # Which Llama layer to extract from / inject into (0-indexed, out of 80)
    target_layer: int = 30

    # Data collection
    chunk_size: int = 1024          # tokens per chunk
    max_chunks_per_doc: int = 8
    data_dir: str = "data/hidden_states"
    output_dir: str = "checkpoints"

    # SVD alignment (Stage 1)
    svd_checkpoint: str = "checkpoints/w_optimal.pt"

    # MLP (Stage 2)
    mlp_hidden_dim: int = 4096
    mlp_num_layers: int = 3         # depth of the residual block
    mlp_dropout: float = 0.0

    # Training
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 10
    warmup_steps: int = 500
    grad_clip: float = 1.0
    lambda_mse: float = 0.1         # weight on MSE term relative to InfoNCE
    infonce_temperature: float = 0.07

    # Validation
    top_k_retrieval: int = 1        # must rank correct pair at position ≤ this

    # Hardware
    # These are *logical* indices within CUDA_VISIBLE_DEVICES, not physical GPU IDs.
    # Set CUDA_VISIBLE_DEVICES in the environment to select which physical GPUs to use.
    # Layout: cuda:3 = Qwen + MLP, cuda:0-2 = Llama (tensor-parallel over 3 cards)
    source_device: str = "cuda:3"
    llama_devices: tuple = (0, 1, 2)   # logical indices for Llama shards
    dtype: str = "bfloat16"
