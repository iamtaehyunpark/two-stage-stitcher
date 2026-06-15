# Two-Stage Latent Stitcher

A cross-family hidden-state translation pipeline that maps the context-rich representations of **Qwen2.5-7B** into the intermediate layers of **DeepSeek-R1-Distill-Llama-70B**, bypassing the expensive prefill step for long documents entirely.

---

## Motivation

Dense 70B models are powerful reasoners but pay a steep cost for long documents: the KV cache for a 128K-token context can consume tens of gigabytes, and the prefill pass alone takes seconds. Meanwhile, retrieval-optimised 7B models like Qwen2.5 process the same context cheaply and accurately.

The core idea: let Qwen read the document and compress it into a single hidden-state vector, then translate that vector into the representation space of the 70B model at a middle layer, as if the 70B had already processed the document up to that point — without it ever doing so.

---

## Architecture

```
Document
   │
   ▼
Qwen2.5-7B-Instruct          (128K context, src_dim = 3584)
   │  last-token hidden state at final layer
   ▼
┌─────────────────────────────────────────┐
│  Stage 1 — SVD Projection  [FROZEN]    │
│  zero-pad 3584 → 8192                  │
│  x_coarse = x_pad @ W_optimal.T        │
│  W_optimal = U @ V^T  (Procrustes)     │
└─────────────────────────────────────────┘
   │  x_coarse ∈ R^8192
   ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Residual MLP  [TRAINED]     │
│  x_final = x_coarse + MLP(x_coarse)    │
│  4 × (LayerNorm → Linear → GELU)       │
│  hidden_dim = 8192                     │
└─────────────────────────────────────────┘
   │  x_final ∈ R^8192
   ▼
Inject at layer 30 of DeepSeek-R1-Distill-Llama-70B
   │
   ▼
Layers 30 → 80 + query tokens → generation
```

### Why two stages?

- **Stage 1 (SVD)** handles the macro coordinate shift between the two model families in closed form. Qwen and DeepSeek use completely different tokenisers, positional encodings, and training distributions, so their representation spaces are geometrically misaligned. The orthogonal Procrustes solution gives the best possible linear mapping in one shot, with no gradient updates needed.
- **Stage 2 (MLP)** repairs the residual non-linear distortions that a linear map cannot capture — dialectical nuances, semantic density gaps, and structural quirks unique to each model family.

---

## Models

| Role | Model | Hidden dim | Context |
|---|---|---|---|
| Source (retrieval) | `Qwen/Qwen2.5-7B-Instruct` | 3584 | 128K |
| Target (reasoning) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 8192 | 128K |

Both models are fully open — no HuggingFace authentication required.

---

## Hardware

Designed for **4× NVIDIA H200 GPUs**:

| GPU(s) | Role |
|---|---|
| `cuda:0`, `cuda:1`, `cuda:2` | DeepSeek-R1-70B shards (tensor parallel) |
| `cuda:3` | Qwen-7B + MLP training |

GPU assignment uses logical indices within `CUDA_VISIBLE_DEVICES`, not physical IDs — see [GPU Selection](#gpu-selection).

---

## Repository Structure

```
two-stage-stitcher/
├── config.py                  # All hyperparameters and paths in one place
├── download_data.py           # Phase 0: download Wikipedia + Gutenberg docs
├── collect_hidden_states.py   # Phase 1: extract paired hidden states
├── svd_alignment.py           # Phase 2: closed-form SVD alignment (Stage 1)
├── stitcher_model.py          # Model definition: SVDProjection + ResidualMLP
├── train_mlp.py               # Phase 3: train residual MLP with InfoNCE + MSE
├── validate.py                # Phase 4: ablation and Top-1 retrieval accuracy
├── inference.py               # Downstream: inject hidden state into DeepSeek
└── run_pipeline.sh            # End-to-end orchestration script
```

---

## Setup

### Environment

This project reuses the `engramtrace-env` virtual environment:

```bash
source /data/tpark45/engramtrace-env/bin/activate
```

All required packages (`torch`, `transformers`, `accelerate`, `scipy`, `numpy`, `tqdm`) are already present in that environment.

### Models

Qwen2.5-7B is already cached at `/data/tpark45/hugginface/hub`. DeepSeek-R1-Distill-Llama-70B will be downloaded automatically on first run to the same location.

```bash
export HF_HOME=/data/tpark45/hugginface
```

---

## Running the Pipeline

### Full run (recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_pipeline.sh
```

This runs all five phases in order. Phases are skipped automatically if their outputs already exist (e.g. docs already downloaded, hidden states already extracted).

### Phase-by-phase

#### Phase 0 — Download data

Downloads 2000 long-form documents (50% Wikipedia, 50% Project Gutenberg) to use as training material.

```bash
python download_data.py --out-dir /data/tpark45/docs --num-docs 2000
```

Documents shorter than 1000 words are skipped. You can adjust the Wikipedia/Gutenberg ratio:

```bash
python download_data.py --out-dir /data/tpark45/docs --num-docs 2000 --wiki-frac 0.3
```

#### Phase 1 — Collect hidden states

For each document, generates progressive cumulative chunks:
`[C₁]`, `[C₁+C₂]`, `[C₁+C₂+C₃]`, …, up to 16 chunks per document.

Each chunk is run through both models and the last-token hidden states are saved as paired `(X, Y)` numpy arrays:

- `X ∈ R^(N × 3584)` — Qwen final layer
- `Y ∈ R^(N × 8192)` — DeepSeek layer 30

```bash
python collect_hidden_states.py \
    --documents /data/tpark45/docs/*.txt \
    --out-dir data/hidden_states/train
```

#### Phase 2 — SVD alignment

Computes the optimal orthogonal projection matrix `W_optimal` from the full training set using the closed-form Procrustes solution:

```
A = X_pad^T @ Y        (cross-covariance, 8192 × 8192)
U, S, V^T = SVD(A)
W_optimal = U @ V^T
```

`X` is zero-padded from 3584 → 8192 before this step.

```bash
python svd_alignment.py
```

Prints Stage-1-only cosine similarity and MSE as a sanity check before MLP training begins.

#### Phase 3 — Train residual MLP

Trains Stage 2 with a dual-objective loss:

```
Loss = InfoNCE(x_final, Y) + λ · MSE(x_final, Y)
```

- **InfoNCE** uses all off-diagonal pairs in the batch as negatives, sharpening the contrastive boundary
- **MSE** anchors the translated vectors to the correct coordinates in DeepSeek's representation space
- **Early stopping** halts training when validation loss stops improving

```bash
python train_mlp.py
```

To resume from an existing checkpoint (e.g. after adding more data):

```bash
python train_mlp.py --resume checkpoints/stitcher_best.pt
```

#### Phase 4 — Validate

Runs ablation on held-out documents, comparing Stage 1 alone vs. the full pipeline:

```bash
python validate.py \
    --val-dir data/hidden_states/val \
    --ckpt checkpoints/stitcher_best.pt
```

Reports:

| Metric | Description |
|---|---|
| MSE | Mean squared error between translated and target hidden states |
| Cosine similarity | Directional alignment (target: → 1.0) |
| Top-1 retrieval accuracy | Does `x_final` rank closest to its correct `y` in the gallery? (target: 1.0) |

Quality gates:
- MSE must decrease from Stage 1 → Full pipeline
- Cosine similarity must increase
- Top-1 accuracy is reported with a warning if below 1.0

---

## Inference

Run inference by injecting `x_final` at layer 30 of DeepSeek, then running layers 30→80 over the query:

```bash
python inference.py \
    --ckpt checkpoints/stitcher_best.pt \
    --document /path/to/document.txt \
    --query "What are the key findings in section 3?"
```

The document never passes through DeepSeek's early layers — only Qwen processes it. This eliminates the KV cache cost for the document context entirely.

---

## Configuration

All parameters live in `config.py`. Key values:

| Parameter | Default | Description |
|---|---|---|
| `target_layer` | 30 | DeepSeek layer to extract from and inject into |
| `chunk_size` | 1024 | Tokens per progressive chunk |
| `max_chunks_per_doc` | 16 | Progressive chunks per document |
| `mlp_hidden_dim` | 8192 | MLP internal width (matches target hidden dim) |
| `mlp_num_layers` | 4 | Residual MLP depth |
| `batch_size` | 256 | Training batch size (more = more InfoNCE negatives) |
| `num_epochs` | 50 | Max training epochs |
| `early_stop_patience` | 7 | Epochs without improvement before stopping |
| `lambda_mse` | 0.5 | MSE weight relative to InfoNCE |
| `infonce_temperature` | 0.05 | Contrastive loss temperature |

---

## GPU Selection

GPU indices in the code are **logical** — they refer to positions within `CUDA_VISIBLE_DEVICES`, not physical device IDs.

```bash
# Use physical GPUs 4, 5, 6, 7
CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_pipeline.sh

# Let the script auto-select 4 GPUs with the most free memory
./run_pipeline.sh

# On SLURM — CUDA_VISIBLE_DEVICES is set automatically, just run:
./run_pipeline.sh
```

The layout within those 4 logical GPUs is always:
- `cuda:0`, `cuda:1`, `cuda:2` → DeepSeek-70B shards
- `cuda:3` → Qwen-7B inference + MLP training
