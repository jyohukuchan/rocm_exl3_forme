#!/usr/bin/env bash
# Build libr4d's r4d.so for EXL3's opt-in backend (see doc/r4d_attention.md).
#
# License boundary: this script clones third-party sources and fetches the
# patch at BUILD TIME on the user's machine. The rocm_exl3 repo itself vendors
# none of their code, patches, or binaries -- only this script and the
# ctypes loader (exllamav3/modules/attention_fn/r4d_attn.py).
#
# Both upstreams are currently unlicensed (all rights reserved): do not copy
# their files into this repo, and do not run their setup/quickstart scripts or
# images -- only the minimal clone + patch + build below.
#
# Usage (run OUTSIDE the repo, e.g. in a scratch dir, inside the ROCm container
# that will load the library -- it must be compiled with the same ROCm/hipcc):
#
#   GFX_ARCH=gfx1201 /path/to/rocm_tools/build_r4d.sh
#   EXL3_R4D_LIB=$PWD/.r4d-build/r4d.so python3 ...
#
# Env:
#   R4D_PIN        libr4d commit (default b9e42ab -- the pin vllm-mxfp4's rx
#                  patches are maintained against)
#   R4D_VLLM_REF   vllm-mxfp4 commit hosting r4d_radiance_extras_rx10.patch
#                  (default d616242). The patch is fetched, never vendored.
#   R4D_WORK       build directory (default ./.r4d-build)
#   OUT            final r4d.so path (default $R4D_WORK/r4d.so)
#   GFX_ARCH       offload arch (default gfx1201)
#   JOBS           parallel compile jobs (default nproc)
set -euo pipefail

R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_VLLM_REF=${R4D_VLLM_REF:-d616242}
R4D_LIB_URL=${R4D_LIB_URL:-https://codeberg.org/StillDeadcode/libr4d.git}
PATCH_URL=${PATCH_URL:-https://raw.githubusercontent.com/GGZ14/vllm-mxfp4/${R4D_VLLM_REF}/r4d_radiance_extras_rx10.patch}
R4D_WORK=${R4D_WORK:-./.r4d-build}
OUT=${OUT:-${R4D_WORK}/r4d.so}
GFX_ARCH=${GFX_ARCH:-gfx1201}
JOBS=${JOBS:-$(nproc)}

SRC="${R4D_WORK}/libr4d"
mkdir -p "${R4D_WORK}"

if [ -d "${SRC}/.git" ]; then
  echo "[build_r4d] reusing ${SRC}"
  git -C "${SRC}" fetch --quiet origin || true
else
  echo "[build_r4d] cloning ${R4D_LIB_URL}"
  git clone --quiet "${R4D_LIB_URL}" "${SRC}"
fi
git -C "${SRC}" checkout --quiet "${R4D_PIN}"

echo "[build_r4d] fetching patch ${PATCH_URL}"
curl -fsSL "${PATCH_URL}" -o "${R4D_WORK}/rx.patch"
if git -C "${SRC}" apply --check "${R4D_WORK}/rx.patch"; then
  git -C "${SRC}" apply "${R4D_WORK}/rx.patch"
  echo "[build_r4d] patch applied cleanly"
else
  echo "[build_r4d] WARNING: patch does not apply cleanly at this pin; building unpatched" >&2
fi

echo "[build_r4d] compiling (${GFX_ARCH}, jobs=${JOBS})"
(
  cd "${SRC}"
  GFX_ARCH="${GFX_ARCH}" JOBS="${JOBS}" OUT="${R4D_WORK}/r4d.so.tmp" ./build.sh
)

# Never overwrite a .so that may be loaded: write temp, then atomic rename.
mv -f "${R4D_WORK}/r4d.so.tmp" "${OUT}"

echo "[build_r4d] checking prefill symbols"
if command -v nm >/dev/null 2>&1; then
  nm -D --defined-only "${OUT}" | grep -E "r4d_attn_prefill_h256_gqa6_(bf16|fp8)kv" \
    || { echo "[build_r4d] ERROR: prefill symbols missing" >&2; exit 1; }
fi

echo "[build_r4d] OK: ${OUT}"
echo "[build_r4d] use: EXL3_R4D_LIB=${OUT} EXL3_R4D_ATTN=bf16 ..."
