#!/bin/bash
# One-shot environment setup for CSC2626 project (local or Compute Canada / Killarney).
# Run on a LOGIN node (needs internet to install wheels and warm asset/CLIP caches):
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

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate 2>/dev/null || true
  unset VIRTUAL_ENV
fi

if command -v module >/dev/null 2>&1; then
  set +u
  module load StdEnv/2023 "python/${PY_VER}" cuda/12.2 2>/dev/null \
    || module load "python/${PY_VER}" 2>/dev/null || true
  set -u
fi

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
pip install -q --upgrade pip

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

# Retag manylinux wheels for Compute Canada wheelhouse compatibility when needed.
install_pypi_wheel() {
  local spec="$1"
  local whl
  PIP_CONFIG_FILE=/dev/null pip download -q --isolated --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --python-version "${PY_TAG}" --implementation cp --abi "cp${PY_TAG}" \
    --no-deps -d "${work_dir}" "${spec}"
  whl="$(ls "${work_dir}"/*.whl | head -n 1)"
  mv "${whl}" "${whl%-*}-linux_x86_64.whl"
  pip install -q --no-deps "${whl%-*}-linux_x86_64.whl"
  rm -f "${work_dir}"/*.whl
}

if ! pip install -q --no-deps sapien==3.0.3 2>/dev/null; then
  install_pypi_wheel "sapien==3.0.3"
fi
if ! pip install -q --no-deps "opencv-python-headless<4.12" 2>/dev/null; then
  install_pypi_wheel "opencv-python-headless<4.12"
fi

grep -viE '^(sapien|mani-skill|opencv-python)' requirements.txt > "${work_dir}/reqs.txt"
pip install -q -r "${work_dir}/reqs.txt" --prefer-binary
pip install -q --no-deps mani-skill==3.0.1

echo "=== Warming ManiSkill3 & CLIP caches for offline compute nodes ==="
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
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
