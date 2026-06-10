#!/bin/bash
#SBATCH --job-name=retune-hide-stats
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --partition=long
#SBATCH --exclusive

# Characterize, independently and with many samples, the two distributions that
# decide whether an NVML clock change hides behind a panel transfer:
#   retune cost tau (host blocking of setApplicationsClocks, biggest swing
#     2040<->240 MHz) and transfer time (cudaMemcpy2DAsync over an nvshmem_ptr
#     P2P pointer -- the panel-broadcast primitive of panel Cholesky / LU),
#     swept over panel width so the elbow (transfer ~ tau) is bracketed.
# Every raw sample is dumped so P(tau > transfer) -- the odds the retune fails
# to hide -- can be computed in post-analysis (analyze.py). Needs
# setApplicationsClocks, so run privileged (same path as test_freq_controller).

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

# Fresh directory per run so no run is ever overwritten.
RUN_DIR="/data/users/sargent/experiments/retune-hide-stats/$(date +%Y%m%d-%H%M%S)-job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

CLEAN_GPUS=$(select_gpus 2)
echo "[$SLURMD_NODENAME] retune/transfer distributions -> $RUN_DIR, clean gpus: $CLEAN_GPUS"

run_privileged --np 2 --gpus "$CLEAN_GPUS" --symmetric-size 4G \
    "$BIN_DIR/retune_latency_hiding" \
        --n=16384 \
        --f-hi=2040 \
        --f-lo=240 \
        --cover-b=8192 \
        --retune-samples=2000 \
        --xfer-samples=300 \
        --b-list=256,1024,2048,3072,4096,8192,14336 \
    | tee "${RUN_DIR}/samples.csv"

reset_gpu_clocks "$CLEAN_GPUS"

deactivate
