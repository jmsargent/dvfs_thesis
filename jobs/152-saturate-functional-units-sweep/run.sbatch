#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40s:4
#SBATCH --nodelist=neptune,uranus
#SBATCH --partition=long
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

BENCH="cholesky"
N_SIZE=32000
TILES=8
DB_PATH="/data/users/sargent/dvfs_thesis/database/cholesky_database.csv"
SCALER="constant"

ALL_FREQS=(2040 1815 1740 1665 1590 1515 1440 1365 1290 1215 1140 1065 990 915 840 795 765 690 615 540 465 390 315 240)

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    FREQS=("${ALL_FREQS[@]:0:12}")
else
    FREQS=("${ALL_FREQS[@]:12}")
fi

CLEAN_GPUS=$(select_gpus 4)
echo "[$SLURMD_NODENAME] task $SLURM_ARRAY_TASK_ID, clean gpus: $CLEAN_GPUS"

for FREQ in "${FREQS[@]}"; do

    RUN_DIR="/data/users/sargent/experiments/saturate-functional-units-l40s/${BENCH}/${SCALER}/${FREQ}"
    mkdir -p "${RUN_DIR}/profile" "${RUN_DIR}/energy"

    mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
    sleep 1

    run_privileged --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
        "$BENCHMARK" \
            --name="$BENCH" \
            -n="$N_SIZE" \
            -t="$TILES" \
            --panel \
            -r=5 \
            --streams=1 \
            --output="${RUN_DIR}/profile" \
            --measure=wait_start_ts,start_ts,end_ts \
            --db="${DB_PATH}" \
            --scaler="${SCALER}" \
            --baseline-freq="${FREQ}"

    stop_monitors "${MONITOR_PIDS[@]}"
    reset_gpu_clocks "$CLEAN_GPUS"
    sleep 2

done

deactivate
