#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

# 1. Paths
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
EXECUTABLE_PATH="${PROJECT_DIR}/mustard/build/cholesky_mustard"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
PROFILER_OUT_BASE="${SLURM_SUBMIT_DIR}/benchmark_out"

# 2. Setup IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

# 3. Start the Monitor on the HOST (Not in the container)
# We use your venv python directly
source "$VENV_PATH/bin/activate"

# We start one monitor per GPU requested
IFS=',' read -ra GPUS <<< "$CLEAN_GPUS"
MONITOR_PIDS=()

for i in "${!GPUS[@]}"; do
    python3 "$PROFILER_PATH" --monitor \
    --gpu-id "${GPUS[$i]}" \
    --interval 0.01 \
    --output "${PROFILER_OUT_BASE}_${i}.csv" &
    MONITOR_PIDS+=($!)
done
deactivate

sleep 2

# 4. Run the Containerized Benchmark
apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 2 \
--bind-to core \
-x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
-x NVSHMEM_BOOTSTRAP=MPI \
-x NVSHMEM_REMOTE_TRANSPORT=none \
-x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
-x UCX_NET_DEVICES=all \
-mca pml ob1 \
-mca btl self,vader \
-mca coll_hcoll_enable 0 \
"$EXECUTABLE_PATH" -n=2400 -t=2 --tiled -r=1 --verbose

# 5. Cleanup the Host Monitors
for pid in "${MONITOR_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
done