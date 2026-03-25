#!/bin/bash
#SBATCH --job-name=lu_mustard
#SBATCH --partition=long
#SBATCH --ntasks=4
#SBATCH --gres=gpu:L4:4
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_job_history/result_%j.out
#SBATCH --nodelist=ganymede

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
EXECUTABLE_PATH="/data/users/sargent/mustard/build/lu_mustard"

# 1. Capture IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

nvidia-smi --id=$CLEAN_GPUS --query-gpu=timestamp,index,utilization.gpu,power.draw,energy.consumption \
    --format=csv -lms 100 > slurm_job_history/result_"${SLURM_JOB_ID}"_gpu_log.csv &

LOGGER_PID=$!

# 2. Run the test
apptainer exec --nv --bind /dev/shm "$CONTAINER_PATH" mpirun -np 4 \
    --bind-to core \
    -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
    -x NVSHMEM_BOOTSTRAP=MPI \
    -x NVSHMEM_REMOTE_TRANSPORT=none \
    -x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
    -x UCX_NET_DEVICES=all \
    -mca pml ob1 \
    -mca btl self,vader \
    -mca coll_hcoll_enable 0 \
    $EXECUTABLE_PATH -n=4800 -t=4 --tiled -r=1 --dot

kill $LOGGER_PID