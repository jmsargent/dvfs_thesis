#!/bin/bash
#SBATCH --job-name=mustard_dvfs_comprehensive
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2
#SBATCH --nodelist=ganymede,europa
#SBATCH --partition=long

source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"
CLEAN_GPUS=$(select_gpus)
EXPERIMENT_ID="$(basename "$SLURM_SUBMIT_DIR")"

# Define Parameter Spaces
BENCHMARKS=("lu" "cholesky")
FREQS=(2040) #1815 1665 1515 1365 1215 1065 915 795 690 540 390 315 240)
TILE_SIZES=(16)
N_SIZE=40000
RUNS=5

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
                    --static-multigpu \
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
                    --static-multigpu \
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
