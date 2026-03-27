#!/bin/bash
#SBATCH --job-name=mustard_dvfs_comprehensive
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=ganymede
#SBATCH --output=mustard_dvfs_all_%j.out
#SBATCH --partition=long

# 1. Paths
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
BIN_DIR="${PROJECT_DIR}/mustard/build"

# 2. Setup IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra GPUS <<< "$CLEAN_GPUS"

source "$VENV_PATH/bin/activate"

# 3. Define Parameter Spaces
BENCHMARKS=("cholesky_mustard" "lu_mustard")
FREQS=(795 2040)
TILE_SIZES=(16 4)
N_SIZE=9600
RUNS=5

# 4. Master Loop
for BENCH in "${BENCHMARKS[@]}"; do
    EXECUTABLE_PATH="${BIN_DIR}/${BENCH}"

    for FREQ in "${FREQS[@]}"; do
        for TILE in "${TILE_SIZES[@]}"; do

            # Metadata for parsing
            EXPERIMENT_ID="BENCH=${BENCH}_FREQ=${FREQ}_TILE=${TILE}_N=${N_SIZE}_R=${RUNS}"
            echo "===================================================="
            echo "START_EXPERIMENT: ${EXPERIMENT_ID}"
            echo "===================================================="

            # --- Step A: Set Frequency ---
            echo "Setting GPUs to ${FREQ}MHz..."
            for i in "${!GPUS[@]}"; do
                sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "${GPUS[$i]}" --set-freq --gpu-freq "$FREQ"
            done

            # --- Step B: Start Monitors ---
            MONITOR_PIDS=()
            TEMP_CSV_BASE="/tmp/job_${SLURM_JOB_ID}_${BENCH}_${FREQ}_${TILE}_gpu"

            for i in "${!GPUS[@]}"; do
                python3 "$PROFILER_PATH" --monitor \
                --gpu-id "${GPUS[$i]}" \
                --interval 0.01 \
                --output "${TEMP_CSV_BASE}_${i}.csv" &
                MONITOR_PIDS+=($!)
            done
            sleep 1

            # --- Step C: Run Benchmark ---
            echo "Running ${BENCH}..."
            apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 4 \
            --bind-to core \
            -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
            -x NVSHMEM_BOOTSTRAP=MPI \
            -x NVSHMEM_REMOTE_TRANSPORT=none \
            -x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
            -x UCX_NET_DEVICES=all \
            -mca pml ob1 \
            -mca btl self,vader \
            -mca coll_hcoll_enable 0 \
            "$EXECUTABLE_PATH" -n="$N_SIZE" -t="$TILE" --tiled -r="$RUNS" --verbose

            # --- Step D: Stop Monitors & Reset ---
            for pid in "${MONITOR_PIDS[@]}"; do
                kill "$pid" 2>/dev/null
                (sleep 0.5 && kill -9 "$pid" 2>/dev/null) &
            done

            for i in "${!GPUS[@]}"; do
                sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "${GPUS[$i]}" --reset-clocks
            done
            sleep 2

            # --- Step E: Flush Results to Log ---
            for i in "${!GPUS[@]}"; do
                # Unique tag for regex parsing: CSV_DATA|BENCH|FREQ|TILE|GPU_ID
                TAG="CSV_DATA|${BENCH}|${FREQ}|${TILE}|${GPUS[$i]}"
                echo "--- BEGIN ${TAG} ---"
                if [ -f "${TEMP_CSV_BASE}_${i}.csv" ]; then
                    cat "${TEMP_CSV_BASE}_${i}.csv"
                    rm "${TEMP_CSV_BASE}_${i}.csv"
                else
                    echo "ERROR: Profiler data not found for ${TAG}"
                fi
                echo "--- END ${TAG} ---"
            done

            echo "END_EXPERIMENT: ${EXPERIMENT_ID}"
            echo "----------------------------------------------------"

        done
    done
done

deactivate