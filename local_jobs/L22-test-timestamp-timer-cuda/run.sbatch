#!/bin/bash
#SBATCH --job-name=test_timer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:1
#SBATCH --nodelist=callisto,io,europa,ganymede
#SBATCH --output=local_jobs/L22-test-timestamp-timer-cuda/result_%j.out

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
BIN="/data/users/sargent/dvfs_thesis/mustard/build/test_timer"

apptainer exec --nv --bind /data "$CONTAINER_PATH" "$BIN"
