#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this repair on a Leonardo login node, not inside a SLURM job." >&2
  exit 2
fi

source scripts/00_setup_env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DATASETS_OFFLINE

# vLLM 0.6.4.post1 and PyTorch 2.5.1 are a matched stack. Their official
# Linux wheels target CUDA 12.1, which is loadable by Leonardo's 12.2 driver.
if python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('vllm') else 1)"; then
  python -m pip install --upgrade --force-reinstall "vllm==0.6.4.post1"
fi

python -m pip install --upgrade --force-reinstall \
  "torch==2.5.1" "torchvision==0.20.1" \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch CUDA build={torch.version.cuda}")
if torch.version.cuda != "12.1":
    raise SystemExit("Expected the CUDA 12.1 PyTorch wheel")
print("GPU stack repaired. CUDA availability is checked again inside the SLURM job.")
PY
