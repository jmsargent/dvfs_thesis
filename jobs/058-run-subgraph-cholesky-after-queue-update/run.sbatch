#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
BIN_DIR="${PROJECT_DIR}/mustard/build"

# 2. Setup IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra GPUS <<< "$CLEAN_GPUS"


BENCH="cholesky_mustard"
EXECUTABLE_PATH="${BIN_DIR}/${BENCH}"

N_SIZE=2400
TILE=4
RUNS=1

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
            compute-sanitizer --tool memcheck --print-limit 20 \
            "$EXECUTABLE_PATH" -n="$N_SIZE" --subgraph -t="$TILE" -r="$RUNS" --verbose