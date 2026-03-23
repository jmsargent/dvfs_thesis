#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
if [ ! -f "$CONTAINER_PATH" ]; then
echo "COULD NOT FIND REMOTE IMAGE PATH" && exit 0
fi

USER="$(whoami)"
EXECUTABLE_PATH="/data/users/${USER}/dvfs_thesis/mustard/build/cholesky_mustard"
if [ ! -f "$EXECUTABLE_PATH" ]; then
echo "COULD NOT FIND EXECUTABLE_PATH :(" && exit 0
fi

PROFILER_PATH="/data/users/${USER}/dvfs_thesis/chens_code/EnergyMonitor.py"
if [ ! -f "$PROFILER_PATH" ]; then
echo "COULD NOT FIND PROFILER_PATH :(" && exit 0
fi

# 1. Capture IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')

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
# 1. Identify Ranks (Use \ to escape $ for container evaluation)
RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

# 2. Lock Clocks (Since you have root)
sudo nvidia-smi -i \$LOCAL_RANK -lgc 1500
trap 'sudo nvidia-smi -i \$LOCAL_RANK -rgc' EXIT

# 3. Run Profiler (Removed the ./ prefix)
python3 \"$PROFILER_PATH\" --monitor \
--gpu-id \"\$LOCAL_RANK\" \
--interval 0.01 \
--output \"\${OUT_FILE}_\${RANK}\" &

PROFILER_PID=\$!
sleep 1

# 4. Run Benchmark (Removed the ./ prefix)
\"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose

kill \$PROFILER_PID
"