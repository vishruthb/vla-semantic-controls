#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$(uv --version 2>/dev/null || true)" != "uv 0.9.0" ]]; then
  echo "Require uv 0.9.0: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! ldconfig -p | grep -q libOSMesa; then
  echo "Missing libOSMesa. Ubuntu 24.04: sudo apt-get install -y libosmesa6 libglfw3 libgl1" >&2
  exit 1
fi

uv sync --frozen
uv run --frozen python src/cache_models.py
uv run --frozen python - <<'PY'
import mujoco
import torch
import lerobot
import libero
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
print(f"lerobot={lerobot.__version__} mujoco={mujoco.__version__} libero={libero.__file__}")
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
PY
