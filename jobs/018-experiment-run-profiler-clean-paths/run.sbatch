#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

# 1. Paths and Validation
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
USER="$(whoami)"
EXECUTABLE_PATH="/data/users/${USER}/dvfs_thesis/mustard/build/cholesky_mustard"
PROFILER_PATH="/data/users/${USER}/dvfs_thesis/chens_code/EnergyMonitor.py"

# Verify files exist on the host
if [ ! -f "$CONTAINER_PATH" ]; then echo "MISSING CONTAINER" && exit 1; fi
if [ ! -f "$EXECUTABLE_PATH" ]; then echo "MISSING EXECUTABLE" && exit 1; fi
if [ ! -f "$PROFILER_PATH" ]; then echo "MISSING PROFILER" && exit 1; fi

# 2. Setup Output and Hardware IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

# Define paths for the container to use
PROFILER_OUT_BASE="${SLURM_SUBMIT_DIR}/benchmark_out"
NVIDIA_PY="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/profilers/Nsight_Systems/target-linux-x64/python/bin/python"

# 3. Execution Block
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
/bin/bash -c "
    # Rank identification (escaped for container-side evaluation)
    RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

    # Start Profiler using absolute path to the internal NVIDIA Python
    # Variables without backslashes are expanded by the host before launch
    \"$NVIDIA_PY\" \"$PROFILER_PATH\" --monitor \
    --gpu-id \"\$LOCAL_RANK\" \
    --interval 0.01 \
    --output \"${PROFILER_OUT_BASE}_\${RANK}.csv\" &

    P_PID=\$!
    sleep 2

    # Run Benchmark
    \"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose

    # Cleanup (check if process still exists before killing)
    if ps -p \$P_PID > /dev/null; then
        kill \$P_PID
    fi
"