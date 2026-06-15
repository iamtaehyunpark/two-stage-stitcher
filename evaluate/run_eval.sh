#!/usr/bin/env bash
# End-to-end evaluation pipeline.
#
# Step 1: Generate QA pairs from held-out docs  (Qwen-72B-AWQ via vLLM)
# Step 2: Run conditions A, B, C               (vLLM + stitcher)
# Step 3: LLM-as-judge scoring                 (Qwen-72B-AWQ via vLLM)
#
# Usage:
#   ./evaluate/run_eval.sh
#   ./evaluate/run_eval.sh --skip 1      # skip QA generation (reuse existing)
#   ./evaluate/run_eval.sh --skip 1,2    # jump straight to judging

set -euo pipefail

export HF_HOME=/data/tpark45/hugginface

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR="${DOCS_DIR:-/data/tpark45/docs}"
NUM_EVAL_DOCS="${NUM_EVAL_DOCS:-30}"
CKPT="${CKPT:-checkpoints/stitcher_best.pt}"
QA_FILE="evaluate/data/qa_pairs.json"
RESULTS_FILE="evaluate/data/condition_results.json"
SCORES_FILE="evaluate/data/scores.json"

# ── Skip list ─────────────────────────────────────────────────────────────────
SKIP=""
if [[ "${1:-}" == "--skip" ]]; then
    SKIP="${2:-}"
    shift 2 || true
fi
should_run() { echo "${SKIP}" | tr ',' '\n' | grep -qx "${1}" && return 1 || return 0; }

# ── Env ───────────────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-/data/tpark45/engramtrace-env}"
source "${VENV_DIR}/bin/activate"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(python3 - <<'EOF'
import subprocess
out = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits"
]).decode()
gpus = sorted(
    [(int(l.split(",")[0]), int(l.split(",")[1].strip()))
     for l in out.strip().splitlines()],
    key=lambda x: -x[1]
)
print(",".join(str(g[0]) for g in gpus[:4]))
EOF
)
    export CUDA_VISIBLE_DEVICES
fi

mkdir -p evaluate/data

echo "============================================================"
echo " CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo " Docs dir             : ${DOCS_DIR}"
echo " Num eval docs        : ${NUM_EVAL_DOCS}"
echo " Stitcher checkpoint  : ${CKPT}"
echo " Skipping steps       : ${SKIP:-none}"
echo "============================================================"

# ── Step 1: Generate QA pairs ─────────────────────────────────────────────────
echo ""
if should_run 1; then
    echo "[Step 1] Generating QA pairs …"
    python evaluate/generate_qa.py \
        --docs-dir "${DOCS_DIR}" \
        --num-docs "${NUM_EVAL_DOCS}" \
        --out "${QA_FILE}"
else
    echo "[Step 1] Skipped — using existing ${QA_FILE}"
fi

# ── Step 2: Run conditions A, B, C ───────────────────────────────────────────
echo ""
if should_run 2; then
    echo "[Step 2] Running conditions A, B, C …"
    python evaluate/run_conditions.py \
        --qa   "${QA_FILE}" \
        --ckpt "${CKPT}" \
        --out  "${RESULTS_FILE}"
else
    echo "[Step 2] Skipped — using existing ${RESULTS_FILE}"
fi

# ── Step 3: Judge ─────────────────────────────────────────────────────────────
echo ""
if should_run 3; then
    echo "[Step 3] LLM-as-judge scoring …"
    python evaluate/judge.py \
        --results "${RESULTS_FILE}" \
        --out     "${SCORES_FILE}"
else
    echo "[Step 3] Skipped"
fi

echo ""
echo "============================================================"
echo " Evaluation complete. Results: ${SCORES_FILE}"
echo "============================================================"
