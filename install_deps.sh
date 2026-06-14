#!/bin/bash
# Install all dependencies for the two-stage stitcher into a local venv.
# Targets the cluster: CUDA 12.6, 4x H200.
# PyTorch cu126 is installed first so everything links against it.

set -e

VENV_DIR="$(dirname "$0")/.venv"

echo "=== Step 1: Create virtual environment ==="
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created ${VENV_DIR}"
else
    echo "  ${VENV_DIR} already exists, reusing"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "  Active Python: $(which python)"

echo ""
echo "=== Step 2: PyTorch with CUDA 12.6 ==="
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== Step 3: ML libraries ==="
pip install \
    transformers \
    accelerate \
    scipy \
    numpy \
    tqdm

echo ""
echo "=== Verifying ==="
python -c "
import importlib, sys, torch
libs = ['torch', 'transformers', 'accelerate', 'scipy', 'numpy', 'tqdm']
ok = True
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f'  OK  {lib}')
    except ImportError as e:
        print(f'  FAIL {lib}: {e}')
        ok = False
print(f'  CUDA available : {torch.cuda.is_available()}')
print(f'  GPU count      : {torch.cuda.device_count()}')
print(f'  PyTorch version: {torch.__version__}')
sys.exit(0 if ok else 1)
"
echo ""
echo "Venv ready. Activate with: source ${VENV_DIR}/bin/activate"
