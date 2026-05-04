#!/bin/bash
#SBATCH --job-name=mustard
#SBATCH --output=local_jobs/L27-cholesky/result_%j.out
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2
#SBATCH --nodelist=io
#SBATCH --partition=long
#NOTSBATCH --exclusive

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

CLEAN_GPUS=$(select_gpus 2)

N_SIZE=40000
TILE=16
RUNS=5
STREAMS=(1 32)

echo "got clean gpus: $CLEAN_GPUS"

SCRIPT_DIR="/data/users/sargent/dvfs_thesis/local_jobs/L27-cholesky"

for BENCH in cholesky; do
    for S in "${STREAMS[@]}"; do
        if [[ "$S" == "1" ]]; then
            LABEL="serial"
        else
            LABEL="concurrent"
        fi

        RUN_DIR="${SCRIPT_DIR}/${BENCH}/${LABEL}"
        mkdir -p "${RUN_DIR}/energy" "${RUN_DIR}/taskoutput"

        echo "starting monitors"
        mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
        sleep 1
        echo "monitors started"

        echo "running $BENCH streams=$S"

        run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
            "$BENCHMARK" \
                --name="$BENCH" \
                -n="$N_SIZE" \
                -t="$TILE" \
                --panel \
                -r="$RUNS" \
                --streams="$S" \
                --verbose \
                --measure=wait_start_ts,start_ts,end_ts \
                --output="${RUN_DIR}/taskoutput/timings"

        echo "stopping monitors"
        stop_monitors "${MONITOR_PIDS[@]}"
        echo "monitors stopped"
    done
done

deactivate
