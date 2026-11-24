#!/bin/bash
# One-shot environment setup for CSC2626 project (Compute Canada / Killarney or local).
# Run on a LOGIN node (needs internet for PyPI wheels + asset/CLIP cache warmup):
#   bash setup.sh [--keep-venv] && source .venv/bin/activate

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

readonly PY_VER="${PY_VER:-3.11}"
readonly PY_TAG="${PY_VER//./}"
readonly VENV_DIR="${REPO_ROOT}/.venv"
keep_venv=0
for arg in "$@"; do
  case "${arg}" in
    --keep-venv) keep_venv=1 ;;
    *) echo "Usage: bash setup.sh [--keep-venv]" >&2; exit 2 ;;
  esac
done

log() { printf '\n=== %s ===\n' "$*"; }

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

log "[1/5] Loading cluster modules"
if command -v module >/dev/null 2>&1; then
  set +u
  module load StdEnv/2023 "python/${PY_VER}" cuda/12.2 2>/dev/null \
    || module load "python/${PY_VER}" 2>/dev/null || true
  set -u
fi

log "[2/5] Creating virtual environment at ${VENV_DIR}"
py_bin="$(command -v "python${PY_VER}" || command -v python3)"
[[ -d "${VENV_DIR}" && "${keep_venv}" -eq 0 ]] && rm -rf "${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download -p "${py_bin}" "${VENV_DIR}"
  else
    "${py_bin}" -m venv "${VENV_DIR}"
  fi
fi

activate_file="${VENV_DIR}/bin/activate"
if command -v module >/dev/null 2>&1 && ! grep -q "CSC2626_MODULE_HOOK" "${activate_file}"; then
  tmp_act="$(mktemp)"
  cat <<EOF > "${tmp_act}"
# CSC2626_MODULE_HOOK -- load cluster modules before activating venv
if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 python/${PY_VER} cuda/12.2 2>/dev/null || module load python/${PY_VER} 2>/dev/null || true
fi
EOF
  cat "${activate_file}" >> "${tmp_act}"
  mv "${tmp_act}" "${activate_file}"
fi

# shellcheck disable=SC1090
source "${activate_file}"
pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip
site_pkgs="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

if [[ "${keep_venv}" -eq 1 ]] && python -c "import mani_skill, sapien, torch, transformers" 2>/dev/null; then
  log "[3-4/5] Reusing existing packages in ${VENV_DIR} (--keep-venv)"
else
  # Stage A: Install core scientific/ML stack from local Compute Canada wheelhouse
  # (--no-index avoids hitting PyPI or compiling C++/Rust sdists for torch/transformers).
  log "[3/5] Installing core ML stack (local wheelhouse first)"
  readonly CORE_PKGS=(
    "numpy>=1.24,<2" scipy matplotlib torch "transformers<5" huggingface_hub
    "gymnasium>=0.29.1" lxml h5py pyyaml tqdm requests defusedxml imageio
    GitPython tabulate ipython absl-py "typing-extensions>=4.12"
  )
  if ! pip install --no-index "${CORE_PKGS[@]}"; then
    echo "Local wheelhouse incomplete; installing core stack with binary wheels..."
    for pkg in "${CORE_PKGS[@]}"; do
      pip install --no-index "${pkg}" 2>/dev/null || pip install --prefer-binary "${pkg}"
    done
  fi

  # Stage B: Download remaining wheels directly from PyPI (--only-binary=:all: so
  # pip NEVER compiles sdists) and retag manylinux wheels for Compute Canada.
  log "[4/5] Installing ManiSkill3, SAPIEN, and pure-Python robotics wheels"
  PIP_CONFIG_FILE=/dev/null pip download --isolated --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform any \
    --python-version "${PY_TAG}" --implementation cp --abi "cp${PY_TAG}" \
    --no-deps -d "${work_dir}" \
    "sapien==3.0.3" "opencv-python-headless<4.12" "mani-skill==3.0.1" \
    pytorch_kinematics_ms pytorch-seed arm-pytorch-utilities \
    transforms3d trimesh dacite "tyro>=0.8.5" docstring-parser typeguard nvidia-ml-py

  for whl in "${work_dir}"/*manylinux*.whl; do
    [[ -e "${whl}" ]] || continue
    mv "${whl}" "${whl%-*}-linux_x86_64.whl"
  done
  pip install --no-index --no-deps "${work_dir}"/*.whl

  if command -v setrpaths.sh >/dev/null 2>&1; then
    for d in "${site_pkgs}/sapien" "${site_pkgs}/cv2" "${site_pkgs}/opencv_python_headless.libs"; do
      [[ -d "${d}" ]] && setrpaths.sh --path "${d}" --add_origin --add_path "${d}" || true
    done
  fi
fi

log "[5/5] Warming ManiSkill3 & CLIP caches for offline compute nodes"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
mkdir -p "${HF_HOME}"
python - <<'EOF'
from envs import COLORS, BIN_COLORS, STACK_VOCAB, SORT_VOCAB, make_env
from models import TextEncoder

for task in ("stack", "sort"):
    env = make_env(task, variant="semantic")
    env.reset(seed=0)
    env.close()

enc = TextEncoder(cache_path="data/clip_cache.pt", mode="clip")
stack_g = [f"stack {b} on {a}, then {c} on {b}" for a in COLORS for b in COLORS for c in COLORS if len({a, b, c}) == 3]
sort_g = [f"put {c1} cube in {b1} bin, then {c2} cube in {b2} bin"
          for c1 in COLORS for c2 in COLORS if c1 != c2
          for b1 in BIN_COLORS for b2 in BIN_COLORS if b1 != b2]
enc.encode(STACK_VOCAB + SORT_VOCAB + stack_g + sort_g)
print(f"Setup verified (text_mode={enc.mode}, cached_phrases={len(enc._cache)}).")
EOF
echo "Done. Activate with: source .venv/bin/activate"
