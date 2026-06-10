#!/bin/bash
#SBATCH --job-name=debug_hang
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:L4:4
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --output=slurm_job_history/result_%j.out
#SBATCH --nodelist=ganymede,europa
#SBATCH --exclusive

source /data/users/sargent/dvfs_thesis/benchscripts/library.sh

BUILD="$BIN_DIR"
CLEAN_GPUS=$(select_gpus 4)

# N=8000 T=8 is the confirmed hang; run LU then Cholesky to see if both hang.
# -v enables wait/signal debug trace; --debug-level WARN adds NVSHMEM warnings.
N=8000
T=8

echo "===== mustard_lu N=$N T=$T ====="
run --np 4 --gpus "$CLEAN_GPUS" --symmetric-size 14G --debug-level WARN -- \
    "$BUILD/lu_mustard" -n="$N" -t="$T" --panel -r=1 --scaler=none -v

echo "===== mustard_chol N=$N T=$T ====="
run --np 4 --gpus "$CLEAN_GPUS" --symmetric-size 14G --debug-level WARN -- \
    "$BUILD/cholesky_mustard" -n="$N" -t="$T" --panel -r=1 --scaler=none -v
