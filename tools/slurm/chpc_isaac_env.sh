#!/bin/bash
set -euo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
  echo "ERROR: PROJECT_ROOT is not set" >&2
  exit 1
fi

if [ -z "${SLURM_TMPDIR:-}" ]; then
  export SLURM_TMPDIR="/tmp/${USER:-$LOGNAME}/slurm_${SLURM_JOB_ID:-manual}"
fi

if [ -z "${REWARDLAB_ISAAC_WORKER_VENV:-}" ]; then
  export REWARDLAB_ISAAC_WORKER_VENV="$PROJECT_ROOT/.venv-isaac"
fi

if [ ! -d "$REWARDLAB_ISAAC_WORKER_VENV" ]; then
  echo "ERROR: Isaac worker venv not found at $REWARDLAB_ISAAC_WORKER_VENV" >&2
  exit 1
fi

PYTHON_VER="$($REWARDLAB_ISAAC_WORKER_VENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PACKAGES="$REWARDLAB_ISAAC_WORKER_VENV/lib/python${PYTHON_VER}/site-packages"

# Prefer full vendor IsaacGym tree when available so gymtorch sources exist.
VENDOR_ISAAC_SRC="$PROJECT_ROOT/tools/vendor/isaacgym/python/isaacgym"
if [ -f "$VENDOR_ISAAC_SRC/_bindings/src/gymtorch/gymtorch.cpp" ]; then
  ISAAC_SRC="$VENDOR_ISAAC_SRC"
else
  ISAAC_SRC="$SITE_PACKAGES/isaacgym"
fi

if [ ! -d "$ISAAC_SRC" ]; then
  echo "ERROR: isaacgym package not found in $SITE_PACKAGES" >&2
  exit 1
fi

ISAAC_STAGE_ROOT="${REWARDLAB_ISAAC_STAGE_ROOT:-$SLURM_TMPDIR/rewardlab_isaac}"
ISAAC_STAGE="$ISAAC_STAGE_ROOT/site-packages"

rm -rf "$ISAAC_STAGE_ROOT"
mkdir -p "$ISAAC_STAGE"
cp -a "$ISAAC_SRC" "$ISAAC_STAGE/"

export PYTHONPATH="$ISAAC_STAGE:${PYTHONPATH:-}"
export ISAAC_BINDINGS="$ISAAC_STAGE/isaacgym/_bindings/linux-x86_64"
export CARB_APP_PATH="$ISAAC_BINDINGS"
export GYM_USD_PLUG_INFO_PATH="$ISAAC_BINDINGS/usd/plugInfo.json"
export LD_LIBRARY_PATH="$ISAAC_BINDINGS:${LD_LIBRARY_PATH:-}"

if [ -z "${REWARDLAB_ISAAC_WORKER_COMMAND:-}" ]; then
  export REWARDLAB_ISAAC_WORKER_COMMAND="$REWARDLAB_ISAAC_WORKER_VENV/bin/python $PROJECT_ROOT/tools/scripts/isaac_worker_py38.py"
fi

if [ ! -f "$ISAAC_STAGE/isaacgym/_bindings/src/gymtorch/gymtorch.cpp" ]; then
  echo "ERROR: staged isaacgym is missing gymtorch.cpp at $ISAAC_STAGE/isaacgym/_bindings/src/gymtorch/gymtorch.cpp" >&2
  echo "Hint: ensure IsaacGym Preview 4 is unpacked at $PROJECT_ROOT/tools/vendor/isaacgym" >&2
  exit 1
fi

echo "[isaac-env] worker venv: $REWARDLAB_ISAAC_WORKER_VENV"
echo "[isaac-env] isaac source: $ISAAC_SRC"
echo "[isaac-env] staged isaacgym: $ISAAC_STAGE/isaacgym"
echo "[isaac-env] worker command: $REWARDLAB_ISAAC_WORKER_COMMAND"
