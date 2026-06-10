#!/bin/bash
#SBATCH --job-name=retune-latency-hiding
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --partition=long
#SBATCH --exclusive

# Can the host latency of an NVML core-clock change be hidden behind a panel
# transfer (cudaMemcpy2DAsync over an nvshmem_ptr P2P pointer -- the same
# primitive panel Cholesky / LU use)? We use the biggest swing the L4 offers,
# 2040 <-> 200 MHz, so tau_call is as large as possible: if even that hides
# (delta_ms ~ 0), hiding is possible. Needs setApplicationsClocks, so run
# privileged (same path as the test_freq_controller jobs).

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

RUN_DIR="/data/users/sargent/experiments/retune-latency-hiding"
mkdir -p "$RUN_DIR"

CLEAN_GPUS=$(select_gpus 2)
echo "[$SLURMD_NODENAME] retune latency hiding, clean gpus: $CLEAN_GPUS"

run_privileged --np 2 --gpus "$CLEAN_GPUS" --symmetric-size 4G \
    "$BIN_DIR/retune_latency_hiding" \
        --n=16384 \
        --b=2048 \
        --f-hi=2040 \
        --f-lo=240 \
        --dir=both \
        --ramp=both \
        --runs=50 \
    | tee "${RUN_DIR}/results.csv"

reset_gpu_clocks "$CLEAN_GPUS"

deactivate
