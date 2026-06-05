#!/bin/bash
#SBATCH --job-name=improved-slackawareness
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=io,ganymede,europa
#SBATCH --partition=long
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

N_SIZE=32000
TILES=8

BENCHES=(lu cholesky)
FREQS=(1515 1440 1365 1290 1215 1140 1065 990 915 840 795 765 690 615 540 465 390 315 240)

function get_db_path() {
    local bench=$1
    if [[ "$bench" == "cholesky" ]]; then
        echo "/data/users/sargent/dvfs_thesis/dbs_2/cholesky_database.csv"
    elif [[ "$bench" == "lu" ]]; then
        echo "/data/users/sargent/dvfs_thesis/dbs_2/lu_database.csv"
    else
        echo "Unknown benchmark: $bench" >&2
        exit 1
    fi
}

ALL_WORK=()
for BENCH in "${BENCHES[@]}"; do
    for FREQ in "${FREQS[@]}"; do
        ALL_WORK+=("${BENCH}:${FREQ}")
    done
done

TOTAL=${#ALL_WORK[@]}
N_TASKS=3
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
if (( END > TOTAL )); then
    END=$TOTAL
fi

CLEAN_GPUS=$(select_gpus 4)
echo "[$SLURMD_NODENAME] task $SLURM_ARRAY_TASK_ID, items $START..$((END-1)) of $TOTAL, clean gpus: $CLEAN_GPUS"

for (( i=START; i<END; i++ )); do
    BENCH="${ALL_WORK[$i]%%:*}"
    FREQ="${ALL_WORK[$i]##*:}"
    DB_PATH=$(get_db_path "$BENCH")

    RUN_DIR="/data/users/sargent/experiments/slackaware-bk-sweep/${BENCH}/combined-slack/${FREQ}"
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
            --scaler="combined-slack" \
            --baseline-freq="${FREQ}" \
            --log-plan

    stop_monitors "${MONITOR_PIDS[@]}"
    reset_gpu_clocks "$CLEAN_GPUS"
    sleep 2
done

deactivate
