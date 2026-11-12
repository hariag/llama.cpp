#!/usr/bin/env bash
# Configure-only checks with fake oneDNN packages; no SYCL SDK or GPU required.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cmake_bin=${CMAKE:-cmake}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/include/oneapi/dnnl" "$tmp/project"
cat > "$tmp/project/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.14)
project(test_dnnl LANGUAGES CXX)
add_library(DNNL::dnnl INTERFACE IMPORTED)
set_target_properties(DNNL::dnnl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TEST_INCLUDE}")
include("${CHECK_MODULE}")
ggml_sycl_validate_dnnl("${TEST_VENDOR}")
CMAKE
write_header() {
    cat > "$tmp/include/oneapi/dnnl/dnnl_config.h" <<HEADER
#define DNNL_RUNTIME_NONE 0u
#define DNNL_RUNTIME_SYCL 512u
#define DNNL_VENDOR_INTEL 1u
#define DNNL_VENDOR_NVIDIA 2u
#define DNNL_GPU_RUNTIME $1
$2
HEADER
}
run_case() {
    local name=$1 expected=$2 runtime=$3 vendor=$4
    shift 4
    local result=0
    "$cmake_bin" -S "$tmp/project" -B "$tmp/build-$name" \
        -DCHECK_MODULE="$root/ggml/src/ggml-sycl/cmake/validate-dnnl.cmake" \
        -DTEST_INCLUDE="$tmp/include" -DTEST_VENDOR="$vendor" -DDNNL_GPU_RUNTIME="$runtime" \
        "$@" > "$tmp/$name.log" 2>&1 || result=$?
    if [[ "$expected" == pass && $result -ne 0 ]] || [[ "$expected" == fail && $result -eq 0 ]]; then
        printf 'unexpected %s outcome for %s\n' "$result" "$name" >&2
        cat "$tmp/$name.log" >&2
        exit 1
    fi
    printf '%s: %s\n' "$name" "$expected"
}
write_header DNNL_RUNTIME_SYCL '#define DNNL_GPU_VENDOR DNNL_VENDOR_INTEL'
run_case legacy-intel pass SYCL INTEL
run_case legacy-dpcpp pass DPCPP INTEL
run_case wrong-target fail SYCL NVIDIA
write_header DNNL_RUNTIME_NONE '#define DNNL_GPU_VENDOR DNNL_VENDOR_INTEL'
run_case cpu-header fail SYCL INTEL
write_header DNNL_RUNTIME_SYCL '/* vendor unavailable */'
run_case missing-vendor fail SYCL INTEL
run_case runtime-none fail NONE INTEL
run_case runtime-ocl fail OCL INTEL
run_case exported-but-no-vendor fail SYCL INTEL -DDNNL_GPU_VENDOR=INTEL
write_header DNNL_RUNTIME_SYCL '#define DNNL_GPU_VENDOR DNNL_VENDOR_INTEL'
run_case exported-vendor pass SYCL INTEL -DDNNL_GPU_VENDOR=INTEL
run_case exported-vendor-mismatch fail SYCL INTEL -DDNNL_GPU_VENDOR=NVIDIA
