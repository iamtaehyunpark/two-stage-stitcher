#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/tpark45/hugginface
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_DIR="$(dirname "$0")/.venv"

# Pass --install as the first argument to create/update the venv first.
if [[ "${1:-}" == "--install" ]]; then
    shift
    bash "$(dirname "$0")/install_deps.sh"
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run: bash install_deps.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Using Python: $(which python)"

# ── Paths (override via env if needed) ───────────────────────────────────────
DOCS_DIR="${DOCS_DIR:-data/raw_documents}"
TRAIN_HS_DIR="${TRAIN_HS_DIR:-data/hidden_states/train}"
VAL_HS_DIR="${VAL_HS_DIR:-data/hidden_states/val}"
SVD_CKPT="${SVD_CKPT:-checkpoints/w_optimal.pt}"
OUT_DIR="${OUT_DIR:-checkpoints}"
STITCHER_CKPT="${OUT_DIR}/stitcher_best.pt"

mkdir -p "${TRAIN_HS_DIR}" "${VAL_HS_DIR}" "${OUT_DIR}"

echo "============================================================"
echo " HF_HOME : ${HF_HOME}"
echo " Docs    : ${DOCS_DIR}"
echo "============================================================"

# ── Phase 1: collect hidden states ───────────────────────────────────────────
echo ""
echo "[Phase 1] Collecting hidden states …"

# Split raw docs 95/5 into train/val by filename
ALL_DOCS=( "${DOCS_DIR}"/*.txt )
TOTAL=${#ALL_DOCS[@]}
VAL_COUNT=$(( TOTAL / 20 ))     # ~5 %
VAL_COUNT=$(( VAL_COUNT < 1 ? 1 : VAL_COUNT ))
TRAIN_COUNT=$(( TOTAL - VAL_COUNT ))

TRAIN_DOCS=( "${ALL_DOCS[@]:0:${TRAIN_COUNT}}" )
VAL_DOCS=( "${ALL_DOCS[@]:${TRAIN_COUNT}}" )

echo "  Total docs: ${TOTAL}  →  train: ${TRAIN_COUNT}  val: ${VAL_COUNT}"

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
