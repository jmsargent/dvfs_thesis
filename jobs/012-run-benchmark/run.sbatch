#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
if [ ! -f "$CONTAINER_PATH" ]; then
echo "COULD NOT FIND REMOTE IMAGE PATH" && exit 0
fi

EXECUTABLE_PATH="/data/users/sargent/dvfs_thesis/mustard/build/cholesky_mustard"
if [ ! -f "$EXECUTABLE_PATH" ]; then
echo "COULD NOT FIND REMOTE EXECUTABLE_PATH :(" && exit 0
fi

# 1. Capture IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

apptainer exec --nv --bind /dev/shm "$CONTAINER_PATH" mpirun -np 2 \
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