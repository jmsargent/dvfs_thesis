#!/bin/bash
#SBATCH --job-name=homogenous-retune-best-parameters
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

# ./extract_freqs.sh ../jobs/077-query-l4-clocks/slurm-172829.out 3 0
FREQS=(2010 1965 1920 1875 1830 1785 1740 1695 1650 1605 1560 1515 1470 1425 1380 1335 1290 1245 1200 1155 1110 1065 1020 975 930 885 840 795 750 705 660 615 570 525 480 435 390 345 300 255 210)
TILES=""
N_SIZE=""

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
}

CLEAN_GPUS=$(select_gpus 4)
echo "got clean gpus: $CLEAN_GPUS"

for BENCH in "${BENCHMARKS[@]}"; do
    for FREQ in "${FREQS[@]}"; do

        set_gpu_freq "$CLEAN_GPUS" "$FREQ"

        set_params "$BENCH"

        RUN_DIR="/data/users/sargent/experiments/homogenous-retune-best-parameters/${BENCH}_${FREQ}"
        
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
                --measure=wait_start_ts,start_ts,end_ts

        stop_monitors "${MONITOR_PIDS[@]}"

        reset_gpu_clocks "$CLEAN_GPUS"
        sleep 2

    done
done

deactivate
