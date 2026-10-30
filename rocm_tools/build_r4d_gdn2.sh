#!/usr/bin/env bash
# Build the exact GDN chunk scan (libr4d issue #4 fix) as r4d_gdn2.so for
# EXL3's opt-in backend (see doc/r4d_gdn.md).
#
# License boundary: this script clones the third-party source and fetches the
# patch at BUILD TIME on the user's machine. The rocm_exl3 repo itself vendors
# none of their code, patches, or binaries -- only this script and the ctypes
# loader (exllamav3/modules/gated_delta_net_fn/r4d_gdn.py).
#
# Both upstreams are currently unlicensed (all rights reserved): do not copy
# their files into this repo, and do not run their setup/quickstart scripts or
# images -- only the minimal clone + patch + build below.
#
# The patch (radiance_gdn2_vs_v050.patch, from vllm-mxfp4) is a diff against
# libr4d v0.5.0's r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip, whose md5 is
# 3c1f87fbc9598b06c0b452f1f6edf09f. Our libr4d pin b9e42ab carries the
# byte-identical scan file (r4d.h there still says 0.4.0), so the patch
# applies directly; the script refuses to build against anything else rather
# than silently mixing versions (same policy as the patch author's build.sh).
#
# Usage (run OUTSIDE the repo, e.g. in a scratch dir, inside the ROCm container
# that will load the library):
#
#   GFX_ARCH=gfx1201 /path/to/rocm_tools/build_r4d_gdn2.sh
#   EXL3_R4D_LIB=$PWD/.r4d-build/r4d.so EXL3_R4D_GDN2_LIB=$PWD/.r4d-build/r4d_gdn2.so ...
#
# Env:
#   R4D_PIN        libr4d commit (default b9e42ab)
#   R4D_VLLM_REF   vllm-mxfp4 commit hosting radiance_gdn2_vs_v050.patch
#                  (default d616242). The patch is fetched, never vendored.
#   R4D_WORK       build directory (default ./.r4d-build, shared with build_r4d.sh)
#   OUT            final r4d_gdn2.so path (default $R4D_WORK/r4d_gdn2.so)
#   GFX_ARCH       offload arch (default gfx1201)
set -euo pipefail

R4D_PIN=${R4D_PIN:-b9e42ab}
R4D_VLLM_REF=${R4D_VLLM_REF:-d616242}
R4D_LIB_URL=${R4D_LIB_URL:-https://codeberg.org/StillDeadcode/libr4d.git}
PATCH_URL=${PATCH_URL:-https://raw.githubusercontent.com/GGZ14/vllm-mxfp4/${R4D_VLLM_REF}/moe-gdn2/radiance_gdn2_vs_v050.patch}
R4D_WORK=${R4D_WORK:-./.r4d-build}
OUT=${OUT:-${R4D_WORK}/r4d_gdn2.so}
GFX_ARCH=${GFX_ARCH:-gfx1201}

# md5s from the patch author's build.sh: the v0.5.0 scan file the patch was
# derived from, r4d_gdn_wmma.h, and the validated patched result.
SCAN=r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip
SCAN_MD5=3c1f87fbc9598b06c0b452f1f6edf09f
WMMA_MD5=dcb10fd659237905e75e19acdbc6c050
PATCHED_MD5=db41219a6486d0262ea8eac007fd7713

SRC="${R4D_WORK}/libr4d"
mkdir -p "${R4D_WORK}"
fail() { echo "[build_gdn2] FAIL: $*" >&2; exit 1; }

if [ -d "${SRC}/.git" ]; then
  echo "[build_gdn2] reusing ${SRC}"
  git -C "${SRC}" fetch --quiet origin || true
else
  echo "[build_gdn2] cloning ${R4D_LIB_URL}"
  git clone --quiet "${R4D_LIB_URL}" "${SRC}"
fi
git -C "${SRC}" checkout --quiet "${R4D_PIN}"

[ "$(md5sum "${SRC}/${SCAN}" | cut -d' ' -f1)" = "${SCAN_MD5}" ] \
  || fail "upstream chunk scan at ${R4D_PIN} differs from the v0.5.0 file the fix was derived from"
[ "$(md5sum "${SRC}/r4d_gdn_wmma.h" | cut -d' ' -f1)" = "${WMMA_MD5}" ] \
  || fail "r4d_gdn_wmma.h at ${R4D_PIN} differs from v0.5.0"

echo "[build_gdn2] (the patch renames the scan to radiance_gdn2.hip, so it"
echo "[build_gdn2] is applied to a temp copy carrying the new name)"
echo "[build_gdn2] fetching patch ${PATCH_URL}"
# No curl/wget in the ROCm container image: fetch with the system python3.
python3 - "${PATCH_URL}" "${R4D_WORK}/gdn2.patch" <<'EOF'
import sys
import urllib.request
url, dest = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"User-Agent": "exl3-build/1.0"})
with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
    f.write(r.read())
EOF
[ -s "${R4D_WORK}/gdn2.patch" ] || fail "patch download is empty"

# Apply the patch to a temp copy carrying the new (renamed) file name.
# `git apply` needs no repo for this; the container's git handles the
# rename as long as the target file exists with matching content.
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
cp "${SRC}/${SCAN}" "${WORK}/radiance_gdn2.hip"
(cd "${WORK}" && git apply --check "${R4D_WORK}/gdn2.patch") \
  || fail "exact-scan patch does not apply to the temp copy"
(cd "${WORK}" && git apply "${R4D_WORK}/gdn2.patch")
[ "$(md5sum "${WORK}/radiance_gdn2.hip" | cut -d' ' -f1)" = "${PATCHED_MD5}" ] \
  || fail "patched scan is not the validated file (got $(md5sum "${WORK}/radiance_gdn2.hip" | cut -d' ' -f1))"

echo "[build_gdn2] compiling (${GFX_ARCH})"
# The patched TU is a pybind module (EXACT template + bindings), like the
# patch author's build; only the raw C scan symbol is used via ctypes.
# shellcheck disable=SC2046
hipcc -O3 -std=c++17 -fPIC -shared --offload-arch="${GFX_ARCH}" \
  -Wno-unused-result -ffp-contract=off -mcumode \
  -I "${SRC}" $(python3 -m pybind11 --includes) \
  "${WORK}/radiance_gdn2.hip" -o "${R4D_WORK}/r4d_gdn2.so.tmp"

# Never overwrite a .so that may be loaded: write temp, then atomic rename.
mv -f "${R4D_WORK}/r4d_gdn2.so.tmp" "${OUT}"

echo "[build_gdn2] checking pybind module symbol"
if command -v nm >/dev/null 2>&1; then
  nm -D --defined-only "${OUT}" | grep -q "PyInit_radiance_gdn2" \
    || fail "radiance_gdn2 module init missing"
  # The exact entry is reached through the pybind surface
  # (gdn_chunk_scan_k128_v128_c64_bf16); the raw C symbol of the stock
  # kernel intentionally does not exist in this module.
fi
python3 - "${OUT}" <<'EOF'
import importlib.util
import sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("radiance_gdn2", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert (m.GDN_HEAD_K, m.GDN_HEAD_V, m.GDN_CHUNK) == (128, 128, 64), "geometry mismatch"
assert callable(m.gdn_chunk_scan_k128_v128_c64_bf16), "exact entry missing"
print("[build_gdn2] module self-check OK: exact entry present, geometry 128/128/64")
EOF

echo "[build_gdn2] OK: ${OUT}"
echo "[build_gdn2] use: EXL3_R4D_GDN=1 EXL3_R4D_GDN_EXACT=1 EXL3_R4D_GDN2_LIB=${OUT} ..."
