#!/bin/bash
#SBATCH --job-name=cholesky_dvfs_4gpu
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --output=cholesky_dvfs_%j.out

# 1. Paths
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
EXECUTABLE_PATH="${PROJECT_DIR}/mustard/build/cholesky_mustard"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"

# 2. Setup IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra GPUS <<< "$CLEAN_GPUS"

# 3. Frequency Configuration (4 GPUs)
# Note: L4 Boost is 2040 MHz, Base is 795 MHz.
# Applying 2040 MHz to all for a Boost-state test.
TARGET_FREQ=2040

source "$VENV_PATH/bin/activate"

echo "--- Step 1: Setting Frequencies to ${TARGET_FREQ}MHz ---"
for i in "${!GPUS[@]}"; do
    sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "${GPUS[$i]}" --set-freq --gpu-freq $TARGET_FREQ
done

echo "--- Step 2: Starting Background Monitors ---"
MONITOR_PIDS=()
TEMP_CSV_BASE="/tmp/job_${SLURM_JOB_ID}_gpu"

for i in "${!GPUS[@]}"; do
    python3 "$PROFILER_PATH" --monitor \
    --gpu-id "${GPUS[$i]}" \
    --interval 0.01 \
    --output "${TEMP_CSV_BASE}_${i}.csv" &
    MONITOR_PIDS+=($!)
done

# Brief sleep to ensure monitors and NVML are initialized
sleep 1

echo "--- Step 3: Running Containerized Benchmark ---"
apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 4 \
--bind-to core \
-x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
-x NVSHMEM_BOOTSTRAP=MPI \
-x NVSHMEM_REMOTE_TRANSPORT=none \
-x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
-x UCX_NET_DEVICES=all \
-mca pml ob1 \
-mca btl self,vader \
-mca coll_hcoll_enable 0 \
"$EXECUTABLE_PATH" -n=4800 -t=16 --tiled -r=1 --verbose

echo "--- Step 4: Cleanup and Reset ---"

# Kill monitors politely then forcefully
for pid in "${MONITOR_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
    (sleep 0.5 && kill -9 "$pid" 2>/dev/null) &
done

# Reset clocks to default
for i in "${!GPUS[@]}"; do
    sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "${GPUS[$i]}" --reset-clocks
done

deactivate

# Ensure filesystem has flushed the CSVs
sleep 1

echo "--- Step 5: Consolidating Results ---"
for i in "${!GPUS[@]}"; do
    echo "--- BEGIN PROFILER DATA GPU ${GPUS[$i]} (Target: ${TARGET_FREQ}MHz) ---"
    if [ -f "${TEMP_CSV_BASE}_${i}.csv" ]; then
        cat "${TEMP_CSV_BASE}_${i}.csv"
        rm "${TEMP_CSV_BASE}_${i}.csv"
    else
        echo "ERROR: Profiler data for GPU ${GPUS[$i]} not found."
    fi
    echo "--- END PROFILER DATA GPU ${GPUS[$i]} ---"
done