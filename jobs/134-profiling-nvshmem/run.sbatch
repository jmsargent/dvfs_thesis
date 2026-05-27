#!/bin/bash
#SBATCH --job-name=
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

SCALER="constant"
TILES=""
N_SIZE=""

# LU_FREQS=(240 465 540 1140 1215 ) 
LU_FREQS=(465 1215) 
CHOLESKY_FREQS=(1065)

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
LATENCY_GPUS=$(echo "$CLEAN_GPUS" | cut -d',' -f1,2)

echo "got clean gpus: $CLEAN_GPUS"

BUILD_DIR="/data/users/chjing/dvfs_thesis/mustard/build"
OUT="/data/users/sargent/experiments/profiling-nvshmem"

echo "=== building custom latency tests ==="
apptainer_run cmake --build "$BUILD_DIR" --target latency_signal latency_tile_pull --parallel 4

echo ""
echo "=== signal latency ==="
mkdir -p "${OUT}/signal_latency"
run --np 2 --gpus "$LATENCY_GPUS" --symmetric-size 1G \
    "$BUILD_DIR/latency_signal" --iter 500 \
    --output "${OUT}/signal_latency/signal_latency.csv"

echo ""
echo "=== tile pull latency (LU: tile=1250 N=20000) ==="
mkdir -p "${OUT}/tile_pull_lu"
run --np 2 --gpus "$LATENCY_GPUS" --symmetric-size 1G \
    "$BUILD_DIR/latency_tile_pull" --tile 1250 --n 20000 --iter 200 \
    --output "${OUT}/tile_pull_lu/tile_pull_lu.csv"

echo ""
echo "=== tile pull latency (Cholesky: tile=600 N=24000) ==="
mkdir -p "${OUT}/tile_pull_cholesky"
run --np 2 --gpus "$LATENCY_GPUS" --symmetric-size 1G \
    "$BUILD_DIR/latency_tile_pull" --tile 600 --n 24000 --iter 200 \
    --output "${OUT}/tile_pull_cholesky/tile_pull_cholesky.csv"

echo ""
echo "=== nsys profiling ==="

for BENCH in "lu" "cholesky"; do

    set_params "$BENCH"

    if [[ "$BENCH" == "lu" ]]; then
        FREQS=("${LU_FREQS[@]}")
    else
        FREQS=("${CHOLESKY_FREQS[@]}")
    fi

    for FREQ in "${FREQS[@]}"; do

        NSYS_OUT="${OUT}/nsys_${BENCH}/${FREQ}"
        mkdir -p "$NSYS_OUT"

        run_privileged --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
            --nsys-output "${NSYS_OUT}/rank" \
            "$BENCHMARK" \
                --name="$BENCH" \
                -n="$N_SIZE" \
                -t="$TILES" \
                --panel \
                -r=1 \
                --streams=1 \
                --db="${DB_PATH}" \
                --scaler="${SCALER}" \
                --baseline-freq="${FREQ}"

        reset_gpu_clocks "$CLEAN_GPUS"
        sleep 2

    done

done

deactivate
