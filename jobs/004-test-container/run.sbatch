#!/bin/bash
#SBATCH --job-name=test_container
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

DEFINITION_FILE="../../containers/container.def"
IMAGE_PATH="../../containers/container.sif"



if [ ! -f "$IMAGE_PATH" ]; then
    echo "image not built yet, building..."
    apptainer build --fakeroot "$IMAGE_PATH" "$DEFINITION_FILE"
fi

apptainer exec --nv --bind /dev/shm "$IMAGE_PATH" echo "Hello World"