#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L40s:4
#SBATCH --partition=long
#SBATCH --nodelist=uranus
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

N_SIZE=64000
TILES=32

BENCHES=(lu cholesky)
FREQ=2520
# FREQS=(1515 1440 1365 1290 1215 1140 1065 990 915 840 795 765 690 615 540 465 390 315 240)

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

CLEAN_GPUS=$(select_gpus 4)
echo "[$SLURMD_NODENAME] running all benchmarks, clean gpus: $CLEAN_GPUS"

for BENCH in "${BENCHES[@]}"; do
    DB_PATH=$(get_db_path "$BENCH")

    RUN_DIR="/data/users/sargent/experiments/syncpoints-bk-sweepL40/${BENCH}/combined-slack/${FREQ}"
    mkdir -p "${RUN_DIR}/profile" "${RUN_DIR}/energy"

    mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
    sleep 1

    run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
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
            --scaler="none" \
            --baseline-freq="${FREQ}" \
            --log-plan

    stop_monitors "${MONITOR_PIDS[@]}"
    sleep 2
done

deactivate