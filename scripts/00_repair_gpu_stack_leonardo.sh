#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this repair on a Leonardo login node, not inside a SLURM job." >&2
  exit 2
fi

source scripts/00_setup_env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DATASETS_OFFLINE

# PyTorch 2.5.1's official CUDA 12.1 wheel is loadable by Leonardo's 12.2
# driver. Install it first; torchvision may temporarily pull the newest NumPy.
python -m pip install --upgrade --force-reinstall \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cu121

# Restore one coherent training stack after pip has resolved the CUDA wheels.
# NumPy <2 is required by vLLM 0.6.4.post1. The old vLLM release and TRL 0.23.1
# are both compatible with this Transformers release and PyTorch 2.5.1.
python -m pip install --upgrade --force-reinstall \
  "numpy==1.26.4" \
  "datasets==4.7.0" \
  "accelerate==1.10.1" \
  "transformers==4.56.2" \
  "trl==0.23.1"

# This quantized-inference kernel can be left behind by newer packages and
# requires torch>=2.7. MolecularIQ does not use it.
python -m pip uninstall -y humming-kernels

# Keep an already-installed evaluation backend on the release matched to
# torch 2.5.1 without asking pip to resolve its broad dependencies again.
if python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm') else 1)"; then
  python -m pip install --upgrade --force-reinstall --no-deps "vllm==0.6.4.post1"
fi

python - <<'PY'
import accelerate
import datasets
import numpy
import torch
import transformers
import trl

print(f"torch={torch.__version__}")
print(f"torch CUDA build={torch.version.cuda}")
print(f"numpy={numpy.__version__}")
print(f"datasets={datasets.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"transformers={transformers.__version__}")
print(f"trl={trl.__version__}")
if torch.version.cuda != "12.1":
    raise SystemExit("Expected the CUDA 12.1 PyTorch wheel")
PY

python -m pip check
echo "GPU and Python dependency stack repaired."
