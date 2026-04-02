#!/bin/bash
#SBATCH --job-name=nvshmem_debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:8                 # Request all 8
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --gres=gpu:L4:4
#SBATCH --time=00:10:00

# Build a map of GPU -> NUMA node
declare -A NUMA_MAP
for GPU_ID in $(seq 0 7); do
    PCI_BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID \
        | sed 's/00000000://' | tr '[:upper:]' '[:lower:]')
    NUMA=$(cat /sys/bus/pci/devices/${PCI_BUS}/numa_node 2>/dev/null || echo -1)
    NUMA_MAP[$GPU_ID]=$NUMA
    echo "GPU$GPU_ID -> NUMA $NUMA"
done

# Group GPUs by NUMA node
declare -A NUMA_GROUPS
for GPU_ID in "${!NUMA_MAP[@]}"; do
    NUMA=${NUMA_MAP[$GPU_ID]}
    NUMA_GROUPS[$NUMA]+="$GPU_ID,"
done

# Pick the NUMA group with most GPUs (should be 4)
BEST_NUMA=""
BEST_COUNT=0
for NUMA in "${!NUMA_GROUPS[@]}"; do
    COUNT=$(echo "${NUMA_GROUPS[$NUMA]}" | tr ',' '\n' | grep -c '[0-9]')
    echo "NUMA $NUMA has $COUNT GPUs: ${NUMA_GROUPS[$NUMA]}"
    if [ "$COUNT" -gt "$BEST_COUNT" ]; then
        BEST_COUNT=$COUNT
        BEST_NUMA=$NUMA
    fi
done

# Take first 4 GPUs from best NUMA node, clean up trailing comma
CLEAN_GPUS=$(echo "${NUMA_GROUPS[$BEST_NUMA]}" | tr ',' '\n' \
    | grep '[0-9]' | head -4 | tr '\n' ',' | sed 's/,$//')

echo "Selected NUMA $BEST_NUMA GPUs: $CLEAN_GPUS"

# Sanity check
GPU_COUNT=$(echo "$CLEAN_GPUS" | tr ',' '\n' | grep -c '[0-9]')
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "ERROR: Could not find 4 GPUs on same NUMA node, got $GPU_COUNT"
    exit 1
fi

timeout 60 apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 4 \
            --bind-to core \
            -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
            -x NVSHMEM_BOOTSTRAP=MPI \
            -x NVSHMEM_REMOTE_TRANSPORT=none \
            -x NVSHMEM_DEBUG=1 \
            -x NVSHMEM_DEBUG_SUBSYS=ALL \
            -x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
            -x UCX_NET_DEVICES=all \
            -mca pml ob1 \
            -mca btl self,vader \
            -mca coll_hcoll_enable 0 \
            compute-sanitizer --tool memcheck \
            "$EXECUTABLE_PATH" -n="$N_SIZE" --subgraph -t="$TILE" -r="$RUNS" --verbose \
            2>&1 | tee debug_out.txt | head -200