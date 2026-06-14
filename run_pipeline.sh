#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/tpark45/hugginface

# ── HuggingFace auth ──────────────────────────────────────────────────────────
# Llama-3.1 is gated. Either run `huggingface-cli login` once, or set HF_TOKEN.
if [ -z "${HF_TOKEN:-}" ]; then
    TOKEN_FILE="${HF_HOME}/token"
    if [ -f "${TOKEN_FILE}" ]; then
        export HF_TOKEN=$(cat "${TOKEN_FILE}")
    else
        echo "WARNING: HF_TOKEN not set and no token found at ${TOKEN_FILE}." >&2
        echo "         Run: huggingface-cli login" >&2
    fi
fi

# ── GPU selection ─────────────────────────────────────────────────────────────
# Set CUDA_VISIBLE_DEVICES to the 4 physical GPU IDs you want to use.
# Inside the code, logical indices are always 0-3:
#   cuda:0, cuda:1, cuda:2  → Llama-70B shards
#   cuda:3                  → Qwen-7B + MLP training
#
# Default: use whichever 4 GPUs are visible (e.g. set by SLURM automatically).
# Override example: CUDA_VISIBLE_DEVICES=4,5,6,7 ./run_pipeline.sh
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Auto-select 4 GPUs with the most free memory
    CUDA_VISIBLE_DEVICES=$(python3 - <<'EOF'
import subprocess, json
out = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits"
]).decode()
gpus = sorted(
    [(int(line.split(",")[0]), int(line.split(",")[1].strip()))
     for line in out.strip().splitlines()],
    key=lambda x: -x[1]   # descending free memory
)
print(",".join(str(g[0]) for g in gpus[:4]))
EOF
)
    export CUDA_VISIBLE_DEVICES
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/data/tpark45/engramtrace-env}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Using Python: $(which python)"

# ── Paths (override via env if needed) ───────────────────────────────────────
DOCS_DIR="${DOCS_DIR:-/data/tpark45/docs}"
NUM_DOCS="${NUM_DOCS:-500}"
TRAIN_HS_DIR="${TRAIN_HS_DIR:-data/hidden_states/train}"
VAL_HS_DIR="${VAL_HS_DIR:-data/hidden_states/val}"
SVD_CKPT="${SVD_CKPT:-checkpoints/w_optimal.pt}"
OUT_DIR="${OUT_DIR:-checkpoints}"
STITCHER_CKPT="${OUT_DIR}/stitcher_best.pt"

mkdir -p "${TRAIN_HS_DIR}" "${VAL_HS_DIR}" "${OUT_DIR}"

echo "============================================================"
echo " HF_HOME              : ${HF_HOME}"
echo " CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo " Docs                 : ${DOCS_DIR}"
echo "============================================================"

# ── Phase 0: download dataset ────────────────────────────────────────────────
echo ""
echo "[Phase 0] Downloading documents …"
EXISTING=$(find "${DOCS_DIR}" -name "*.txt" 2>/dev/null | wc -l)
if [ "${EXISTING}" -ge "${NUM_DOCS}" ]; then
    echo "  ${EXISTING} docs already present in ${DOCS_DIR}, skipping download"
else
    python download_data.py \
        --out-dir "${DOCS_DIR}" \
        --num-docs "${NUM_DOCS}"
fi

# ── Phase 1: collect hidden states ───────────────────────────────────────────
echo ""
echo "[Phase 1] Collecting hidden states …"

# Split raw docs 95/5 into train/val by filename.
# When there are too few docs to split, all go to train and val reuses them.
ALL_DOCS=( "${DOCS_DIR}"/*.txt )
TOTAL=${#ALL_DOCS[@]}

if [ "${TOTAL}" -lt 2 ]; then
    TRAIN_DOCS=( "${ALL_DOCS[@]}" )
    VAL_DOCS=( "${ALL_DOCS[@]}" )
    echo "  Total docs: ${TOTAL}  →  train: ${TOTAL}  val: ${TOTAL} (too few to split)"
else
    VAL_COUNT=$(( TOTAL / 20 ))
    VAL_COUNT=$(( VAL_COUNT < 1 ? 1 : VAL_COUNT ))
    TRAIN_COUNT=$(( TOTAL - VAL_COUNT ))
    TRAIN_DOCS=( "${ALL_DOCS[@]:0:${TRAIN_COUNT}}" )
    VAL_DOCS=( "${ALL_DOCS[@]:${TRAIN_COUNT}}" )
    echo "  Total docs: ${TOTAL}  →  train: ${TRAIN_COUNT}  val: ${VAL_COUNT}"
fi

python collect_hidden_states.py \
    --documents "${TRAIN_DOCS[@]}" \
    --out-dir "${TRAIN_HS_DIR}"

python collect_hidden_states.py \
    --documents "${VAL_DOCS[@]}" \
    --out-dir "${VAL_HS_DIR}"

# ── Phase 2: SVD alignment ────────────────────────────────────────────────────
echo ""
echo "[Phase 2] Computing SVD alignment …"
python svd_alignment.py \
    --data-dir "${TRAIN_HS_DIR}" \
    --out "${SVD_CKPT}"

# ── Phase 3: train residual MLP ───────────────────────────────────────────────
echo ""
echo "[Phase 3] Training residual MLP …"
python train_mlp.py \
    --data-dir "${TRAIN_HS_DIR}" \
    --svd-ckpt "${SVD_CKPT}" \
    --out-dir "${OUT_DIR}"

# ── Phase 4: ablation validation ─────────────────────────────────────────────
echo ""
echo "[Phase 4] Running ablation validation …"
python validate.py \
    --val-dir "${VAL_HS_DIR}" \
    --ckpt "${STITCHER_CKPT}"

echo ""
echo "============================================================"
echo " Pipeline complete. Checkpoint: ${STITCHER_CKPT}"
echo "============================================================"
