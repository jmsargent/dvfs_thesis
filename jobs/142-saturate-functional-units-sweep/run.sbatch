#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=io,ganymede,europa
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

if [[ "${1:-}" == "--node-worker" ]]; then
    shift
    FREQS=("$@")

    CLEAN_GPUS=$(select_gpus 4)
    echo "[$SLURMD_NODENAME] clean gpus: $CLEAN_GPUS"

    for FREQ in "${FREQS[@]}"; do

        RUN_DIR="/data/users/sargent/experiments/sweep-big-kernels/${BENCH}/${SCALER}/${FREQ}"
        mkdir -p "${RUN_DIR}/profile" "${RUN_DIR}/energy"

        mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
        sleep 1

        run_privileged --np 4 --gpus "$CLEAN_GPUS" --symmetric-size 14G \
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

    exit 0
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

HALF=$(( ${#ALL_FREQS[@]} / 2 ))
FREQS_NODE0=("${ALL_FREQS[@]:0:$HALF}")
FREQS_NODE1=("${ALL_FREQS[@]:$HALF}")

echo "Node 0: $NODE0 -> freqs ${FREQS_NODE0[*]}"
echo "Node 1: $NODE1 -> freqs ${FREQS_NODE1[*]}"

srun --nodes=1 --ntasks=4 --nodelist="$NODE0" "$0" --node-worker "${FREQS_NODE0[@]}" &
PID0=$!

srun --nodes=1 --ntasks=4 --nodelist="$NODE1" "$0" --node-worker "${FREQS_NODE1[@]}" &
PID1=$!

wait $PID0 $PID1
