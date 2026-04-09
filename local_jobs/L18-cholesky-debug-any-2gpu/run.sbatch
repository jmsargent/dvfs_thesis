#!/bin/bash
#SBATCH --job-name=mustard
#SBATCH --output=local_jobs/L18-cholesky-debug-any-2gpu/result_%j.out
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
OTHER_USER="sargent"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
BIN_DIR="${PROJECT_DIR}/mustard/build"

# ── Topology recon ───────────────────────────────────────────────────────────
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
echo ""
echo "=== NVLink / PCIe topology ==="
nvidia-smi topo -m
echo ""

# ── NUMA check: verify all 2 assigned GPUs are on the same NUMA node ─────────
# On callisto: NUMA0=GPUs 0-3, NUMA1=GPUs 4-7
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
ALL_GPUS=$(echo "$GPU_LIST" | tr -cd '0-9,')
IFS=',' read -ra ALL_GPU_ARR <<< "$ALL_GPUS"

NUMA0=() NUMA1=()
for gpu in "${ALL_GPU_ARR[@]}"; do
    [ "$gpu" -lt 4 ] && NUMA0+=("$gpu") || NUMA1+=("$gpu")
done

if [ ${#NUMA0[@]} -ge 2 ]; then
    CLEAN_GPUS=$(IFS=','; echo "${NUMA0[*]:0:2}")
elif [ ${#NUMA1[@]} -ge 2 ]; then
    CLEAN_GPUS=$(IFS=','; echo "${NUMA1[*]:0:2}")
else
    echo "ERROR: Cannot get 2 GPUs from one NUMA node. Got: $ALL_GPUS"; exit 1
fi

echo "All assigned GPUs: $ALL_GPUS"
echo "Using GPUs (single NUMA): $CLEAN_GPUS"
echo ""

BENCH="cholesky_mustard"
EXECUTABLE_PATH="${BIN_DIR}/${BENCH}"

N_SIZE=2400
TILE=8
RUNS=1

NVSHMEM_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib
CONTAINER_LD=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/lib64:/.singularity.d/libs

# ── MPI smoke test (no NVSHMEM) ──────────────────────────────────────────────
echo "=== MPI smoke test ==="
apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" \
    mpirun -np 2 --bind-to core \
    -mca pml ob1 -mca btl self,vader -mca coll_hcoll_enable 0 \
    hostname
echo ""

# ── Main run with NVSHMEM debug ──────────────────────────────────────────────
echo "=== Main run ==="
APPTAINERENV_LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CONTAINER_LD}" \
timeout 300 apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 2 \
            --bind-to core \
            -x LD_LIBRARY_PATH \
            -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
            -x NVSHMEM_BOOTSTRAP=MPI \
            -x NVSHMEM_REMOTE_TRANSPORT=none \
            -x NVSHMEM_DISABLE_CUDA_VMM=1 \
            -x NVSHMEM_DEBUG=INFO \
            -x NVSHMEM_DEBUG_FILE="${PROJECT_DIR}/local_jobs/L17-cholesky-debug-callisto-2gpu/nvshmem_debug" \
            -x NVSHMEM_DISABLE_GDRCOPY=1 \
            -x UCX_TLS=sm,cuda_copy,self \
            -x UCX_NET_DEVICES=all \
            -mca pml ob1 \
            -mca btl self,vader \
            -mca coll_hcoll_enable 0 \
            "$EXECUTABLE_PATH" -n="$N_SIZE" -t="$TILE" --static-multigpu -r="$RUNS" --verbose --verify

echo ""
echo "=== NVSHMEM debug logs ==="
for f in "${PROJECT_DIR}/local_jobs/L17-cholesky-debug-callisto-2gpu"/nvshmem_debug.*; do
    echo "--- $f ---"
    cat "$f" 2>/dev/null
done
