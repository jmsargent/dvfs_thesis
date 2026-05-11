#!/bin/bash
#SBATCH --job-name=retune-on-syncpoints
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede,europa,io
#SBATCH --partition=long
#SBATCH --exclusive


source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

# Define Parameter Spaces
BENCHMARKS=("lu" "cholesky")
GOALS=("energy" "edp" "ed2p" "e2dp")

TILES=""
N_SIZE=""

function set_params() {
    local bench=$1
    if [[ "$bench" == "cholesky" ]]; then
        N_SIZE=24000
        TILES=40
        DB_PATH="/data/users/sargent/dvfs_thesis/database/cholesky_database.csv"
    elif [[ "$bench" == "lu" ]]; then
        N_SIZE=20000
        TILES=16
        DB_PATH="/data/users/sargent/dvfs_thesis/database/lu_database.csv"
    else
        echo "Unknown benchmark: $bench" >&2
        exit 1
    fi
}

CLEAN_GPUS=$(select_gpus 4)
echo "got clean gpus: $CLEAN_GPUS"

for BENCH in "${BENCHMARKS[@]}"; do

        set_params "$BENCH"

        for GOAL in "${GOALS[@]}"; do

                RUN_DIR="/data/users/sargent/experiments/retune_on_sync2/${BENCH}/${GOAL}"

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
                        --log-plan \
                        --db="${DB_PATH}" \
                        --goal="$GOAL"

                stop_monitors "${MONITOR_PIDS[@]}"

                reset_gpu_clocks "$CLEAN_GPUS"
                sleep 2

        done
done

deactivate
