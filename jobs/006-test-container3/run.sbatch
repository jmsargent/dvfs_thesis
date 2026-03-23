#!/bin/bash
#SBATCH --job-name=test_container
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

USERNAME="$(whoami)"

DEFINITION_FILE="/data/users/$USERNAME/dvfs_thesis/containers/container.def"
IMAGE_PATH="/data/users/$USERNAME/dvfs_thesis/containers/container.sif"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "image not built yet, building..."
    apptainer build "$IMAGE_PATH" "$DEFINITION_FILE"
fi

if [ ! -f "$DEFINITION_FILE" ]; then
    echo "Could not find image, searching"
    echo "path to dvfs_thesis=$(find "/data/users/$USERNAME" -type d -name "dvfs_thesis")"
fi

apptainer exec --nv --bind /dev/shm "$IMAGE_PATH" echo "Hello World"