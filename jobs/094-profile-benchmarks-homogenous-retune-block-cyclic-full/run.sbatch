#!/bin/bash
#SBATCH --job-name=mustard_dvfs_comprehensive
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede,europa
#SBATCH --partition=long
#SBATCH --exclusive


source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"
CLEAN_GPUS=$(select_gpus)
EXPERIMENT_ID="$(basename "$SLURM_SUBMIT_DIR")"

# Define Parameter Spaces
BENCHMARKS=("lu" "cholesky")
FREQS=(2040 1950 1860 1770 1680 1590 1500 1410 1320 1230 1140 1050 960 870 795 735 645 555 465 375 285 240)
TILE_SIZES=(16)
N_SIZE=40000
RUNS=3

for BENCH in "${BENCHMARKS[@]}"; do
    for FREQ in "${FREQS[@]}"; do
        for TILE in "${TILE_SIZES[@]}"; do

            set_gpu_freq "$CLEAN_GPUS" "$FREQ"

            RUN_DIR="/data/users/sargent/experiments/${EXPERIMENT_ID}/concurrent/${BENCH}_${FREQ}"
            mkdir -p "${RUN_DIR}/profile" "${RUN_DIR}/energy"

            mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
            sleep 1

            run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
                "$BENCHMARK" \
                    --name="$BENCH" \
                    -n="$N_SIZE" \
                    -t="$TILE" \
                    --panel \
                    -r="$RUNS" \
                    --streams=32 \
                    --output="${RUN_DIR}/profile/timings" \
                    --measure=wait_start_ts,start_ts,end_ts

            stop_monitors "${MONITOR_PIDS[@]}"

            reset_gpu_clocks "$CLEAN_GPUS"
            sleep 2

        done
    done
done

for BENCH in "${BENCHMARKS[@]}"; do
    for FREQ in "${FREQS[@]}"; do
        for TILE in "${TILE_SIZES[@]}"; do

            set_gpu_freq "$CLEAN_GPUS" "$FREQ"

            RUN_DIR="/data/users/sargent/experiments/${EXPERIMENT_ID}/serial/${BENCH}_${FREQ}"
            mkdir -p "${RUN_DIR}/profile" "${RUN_DIR}/energy"

            mapfile -t MONITOR_PIDS < <(start_monitors "${RUN_DIR}/energy/gpu" "$CLEAN_GPUS")
            sleep 1

            run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
                "$BENCHMARK" \
                    --name="$BENCH" \
                    -n="$N_SIZE" \
                    -t="$TILE" \
                    --panel \
                    -r="$RUNS" \
                    --streams=1 \
                    --output="${RUN_DIR}/profile/timings" \
                    --measure=wait_start_ts,start_ts,end_ts

            stop_monitors "${MONITOR_PIDS[@]}"

            reset_gpu_clocks "$CLEAN_GPUS"
            sleep 2

        done
    done
done

deactivate
