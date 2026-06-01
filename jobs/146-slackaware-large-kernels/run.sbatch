#!/bin/bash
#SBATCH --job-name=improved-slackawareness
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --partition=long
#SBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

umask 000

# Define Parameter Spaces
BENCHMARKS=("lu" "cholesky")
SCALERS=("combined-slack")

N_SIZE=32000
TILES=8
FREQ=""

function set_params() {
    local bench=$1
    if [[ "$bench" == "cholesky" ]]; then
        DB_PATH="/data/users/sargent/dvfs_thesis/dbs_2/cholesky_database.csv"
        FREQ=1365
    elif [[ "$bench" == "lu" ]]; then
        FREQ=1440
        DB_PATH="/data/users/sargent/dvfs_thesis/dbs_2/lu_database.csv"
    else
        echo "Unknown benchmark: $bench" >&2
        exit 1
    fi
}

CLEAN_GPUS=$(select_gpus 4)

echo "got clean gpus: $CLEAN_GPUS"

for BENCH in "${BENCHMARKS[@]}"; do

    set_params "$BENCH"

    for SCALER in "${SCALERS[@]}"; do

            RUN_DIR="/data/users/sargent/experiments/slackaware-bench-big-kernel/${BENCH}/${SCALER}/${FREQ}"

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
                    --baseline-freq="${FREQ}" \
                    --log-plan 

            stop_monitors "${MONITOR_PIDS[@]}"

            reset_gpu_clocks "$CLEAN_GPUS"
            sleep 2

    done
done

deactivate
