#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
EXECUTABLE_PATH="${PROJECT_DIR}/mustard/build/cholesky_mustard"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
PROFILER_OUT_BASE="${SLURM_SUBMIT_DIR}/benchmark_out"

# The path to the only Python that exists in your container
NSIGHT_PY_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/profilers/Nsight_Systems/target-linux-x64/python/bin"

GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

apptainer exec --nv --bind /dev/shm,/data,"$VENV_PATH" "$CONTAINER_PATH" mpirun -np 2 \
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
    # 1. Manually set up the environment since 'source' is failing to find a base python
    # This puts your venv first, then the Nsight Python second
    export PATH=\"$VENV_PATH/bin:$NSIGHT_PY_DIR:\$PATH\"

    # 2. Identify Ranks
    RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

    # 3. Start Profiler (using 'python' because Nsight binary is named 'python')
    python \"$PROFILER_PATH\" --monitor \
    --gpu-id \"\$LOCAL_RANK\" \
    --interval 0.01 \
    --output \"${PROFILER_OUT_BASE}_\${RANK}.csv\" &

    P_PID=\$!
    sleep 2

    # 4. Run Benchmark
    \"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose

    # 5. Cleanup
    kill \$P_PID
"