#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 --np N --gpus CUDA_VISIBLE_DEVICES --symmetric-size SIZE [--debug-file PATH] -- EXECUTABLE [ARGS...]"
    echo ""
    echo "Required:"
    echo "  --np N                  Number of MPI processes"
    echo "  --gpus LIST             CUDA_VISIBLE_DEVICES value (e.g. 0,1)"
    echo "  --symmetric-size SIZE   NVSHMEM_SYMMETRIC_SIZE value (e.g. 14G)"
    echo ""
    echo "Optional:"
    echo "  --debug-file PATH       NVSHMEM_DEBUG_FILE path"
    exit 1
}

NP=""
GPUS=""
SYMMETRIC_SIZE=""
DEBUG_FILE=""
DEBUG_LEVEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)             NP="$2";            shift 2 ;;
        --gpus)           GPUS="$2";          shift 2 ;;
        --symmetric-size) SYMMETRIC_SIZE="$2"; shift 2 ;;
        --debug-file)     DEBUG_FILE="$2";    shift 2 ;;
        --debug-level)    DEBUG_LEVEL="$2";   shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

[[ -z "$NP" ]]             && { echo "ERROR: --np required";             usage; }
[[ -z "$GPUS" ]]           && { echo "ERROR: --gpus required";           usage; }
[[ -z "$SYMMETRIC_SIZE" ]] && { echo "ERROR: --symmetric-size required"; usage; }
[[ $# -eq 0 ]]             && { echo "ERROR: executable required";       usage; }

NVSHMEM_ARGS=(
    -x NVSHMEM_BOOTSTRAP=MPI
    -x NVSHMEM_REMOTE_TRANSPORT=none
    -x NVSHMEM_DISABLE_CUDA_VMM=1
    -x NVSHMEM_SYMMETRIC_SIZE="$SYMMETRIC_SIZE"
    -x NVSHMEM_DISABLE_GDRCOPY=1
)
[[ -n "$DEBUG_FILE" ]]  && NVSHMEM_ARGS+=(-x NVSHMEM_DEBUG_FILE="$DEBUG_FILE")
[[ -n "$DEBUG_LEVEL" ]] && NVSHMEM_ARGS+=(-x NVSHMEM_DEBUG="$DEBUG_LEVEL")

mpirun -np "$NP" \
    --bind-to core \
    -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES="$GPUS" \
    "${NVSHMEM_ARGS[@]}" \
    -x UCX_TLS=sm,cuda_copy,self \
    -x UCX_NET_DEVICES=all \
    -mca pml ob1 \
    -mca btl self,vader \
    -mca coll_hcoll_enable 0 \
    "$@"
