#!/bin/bash
#SBATCH --job-name=test-freq-controller
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --partition=long
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"

umask 000


CLEAN_GPUS=$(select_gpus 4)
echo "got clean gpus: $CLEAN_GPUS"

run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 1G \
    "$BIN_DIR/test_freq_controller" 240 /data/users/sargent/dvfs_thesis/jobs/117-test-setting-freq

reset_gpu_clocks "$CLEAN_GPUS"
