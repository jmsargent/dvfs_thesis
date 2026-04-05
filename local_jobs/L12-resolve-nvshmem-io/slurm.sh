#!/bin/bash
#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=1
#SBATCH --output=local_jobs/L12-resolve-nvshmem-io/result_%j.out
#SBATCH --nodelist=io

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"

apptainer exec --nv "$CONTAINER_PATH" bash -c "
    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib:\$LD_LIBRARY_PATH
    ldd /data/users/sargent/dvfs_thesis/mustard/build/cholesky_mustard | grep nvshmem
"