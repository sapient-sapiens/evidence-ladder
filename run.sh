#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: run.sh <input_pdf_dir> <output_path>}"
output_path="${2:?usage: run.sh <input_pdf_dir> <output_path>}"

# Docker --read-only: keep OCR cache on tmpfs, not under /app.
export MIB_OCR_CACHE="${MIB_OCR_CACHE:-/tmp/mib_ocr_cache}"

# Parallelism is at the packet level. Letting BLAS/OpenMP/ONNX each open their
# own thread pool inside every worker oversubscribes the four scoring vCPUs and
# costs far more than it wins.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python3 /app/solution.py "$input_dir" "$output_path"
