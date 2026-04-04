#!/bin/bash
#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=1
#SBATCH --output=local_jobs/L8-check-io-topography/result_%j.out
#SBATCH --nodelist=callisto

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
apptainer exec --nv "$CONTAINER_PATH" bash -c "
    echo '=== find libnvshmem ==='
    find / -name 'libnvshmem*' 2>/dev/null

    echo '=== LD_LIBRARY_PATH ==='
    echo \$LD_LIBRARY_PATH

    echo '=== ldd on binary ==='
    ldd '$EXECUTABLE_PATH' | grep -i nvshmem
"
