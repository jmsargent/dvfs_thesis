#!/bin/bash
#SBATCH --job-name=test_container
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

IMAGE_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "COULD NOT FIND REMOTE IMAGE PATH" && exit 0
fi

apptainer exec --nv --bind /dev/shm "$IMAGE_PATH" echo "Hello World"