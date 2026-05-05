#!/bin/bash
#SBATCH --job-name=find-params
#SBATCH --output=local_jobs/L28-find-parameters-to-use/result_%j.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:5
#SBATCH --nodelist=ganymede
#SBATCH --nodes=1
#SBATCH --partition=long


source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"


SCRIPT_DIR="/data/users/sargent/dvfs_thesis/local_jobs/L29-Best-Benchmarksettings-details"


N_SIZE=""
TILES=""
OUTPUT_BASE=""


# take lu / cholesky as argument and set N,TILES,TASK_OUTPUT,ENERGY_OUTPUT accordingly
function set_params() {
    local bench=$1
    if [[ "$bench" == "cholesky" ]]; then
        N_SIZE=24000
        TILES=40
    elif [[ "$bench" == "lu" ]]; then
        N_SIZE=20000
        TILES=16
    else
        echo "Unknown benchmark: $bench" >&2
        exit 1
    fi
    OUTPUT_BASE="${SCRIPT_DIR}/${bench}"
}

CLEAN_GPUS=$(select_gpus 4)
echo "got clean gpus: $CLEAN_GPUS"

for BENCH in cholesky lu; do

        set_params "$BENCH"
        ENERGY_OUTPUT="$OUTPUT_BASE/energy"
        PROGRAM_OUTPUT="$OUTPUT_BASE/out"

        mkdir -p "$ENERGY_OUTPUT" "$PROGRAM_OUTPUT"

        echo "starting monitors"
        mapfile -t MONITOR_PIDS < <(start_monitors "$ENERGY_OUTPUT" "$CLEAN_GPUS")
        sleep 1
        echo "monitors started"

        echo "running $BENCH"

        run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
            "$BENCHMARK" \
                --name="$BENCH" \
                -n="$N_SIZE" \
                -t="$TILES" \
                --panel \
                -r=5 \
                --streams=1 \
                --verbose \
                --measure=wait_start_ts,start_ts,end_ts \
                --output="$PROGRAM_OUTPUT"

        echo "stopping monitors"
        stop_monitors "${MONITOR_PIDS[@]}"
        echo "monitors stopped"
done

deactivate
