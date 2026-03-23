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
    # Source the venv inside the container
    source \"$VENV_PATH/bin/activate\"

    RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

    # Start Profiler
    python3 \"$PROFILER_PATH\" --monitor \
    --gpu-id \"\$LOCAL_RANK\" \
    --interval 0.01 \
    --output \"${PROFILER_OUT_BASE}_\${RANK}.csv\" &

    P_PID=\$!
    sleep 2

    # Run Benchmark
    \"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose

    # Cleanup
    kill \$P_PID
"