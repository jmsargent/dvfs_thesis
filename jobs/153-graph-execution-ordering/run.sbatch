#!/bin/bash
#SBATCH --job-name=graph-ordering
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:3
#SBATCH --nodelist=ganymede,io,europa
#SBATCH --partition=long
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

CLEAN_GPUS=$(select_gpus 3)
echo "[$SLURMD_NODENAME] clean gpus: $CLEAN_GPUS"
FIRST_GPU=$(echo "$CLEAN_GPUS" | cut -d',' -f1)

RUN_DIR="/data/users/sargent/experiments/cuda-scheduler-ordering"
mkdir -p "${RUN_DIR}/creation_order" "${RUN_DIR}/staggered"

# Experiment 1: creation_order — single PE, vary node count
for NODES in 4 8 16; do
    run_privileged --np 1 --gpus "$FIRST_GPU" --symmetric-size 256M -- \
        "${BIN_DIR}/cuda_scheduler_ordering" \
            --experiment=creation_order \
            --nodes="$NODES" \
            --runs=5 \
            --spin-ms=10 \
        > "${RUN_DIR}/creation_order/nodes_${NODES}.csv"
done

# Experiment 2: staggered — 3 PEs, runs both normal and inverse trials back-to-back
run_privileged --np 3 --gpus "$CLEAN_GPUS" --symmetric-size 256M -- \
    "${BIN_DIR}/cuda_scheduler_ordering" \
        --experiment=staggered \
        --runs=10 \
        --long-ms=200 \
        --short-ms=50 \
        --work-ms=10 \
    > "${RUN_DIR}/staggered/results.csv"

deactivate
