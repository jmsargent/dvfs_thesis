#!/bin/bash
#SBATCH --job-name=mustard_dvfs_comprehensive
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:4
#SBATCH --nodelist=europa
#SBATCH --partition=long
#SBATCH --exclusive

# 1. Paths
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
BIN_DIR="${PROJECT_DIR}/mustard/build"
ENERGY_DIR="${SLURM_SUBMIT_DIR}/energy_measurements"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/output"
mkdir -p "$ENERGY_DIR" "$OUTPUT_DIR"

# 2. Setup IDs — pick 4 GPUs from the same NUMA node
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
ALL_GPUS=$(echo "$GPU_LIST" | tr -cd '0-9,')
IFS=',' read -ra ALL_GPU_ARR <<< "$ALL_GPUS"

NUMA0=() NUMA1=()
for gpu in "${ALL_GPU_ARR[@]}"; do
    [ "$gpu" -lt 4 ] && NUMA0+=("$gpu") || NUMA1+=("$gpu")
done

if [ ${#NUMA0[@]} -ge 4 ]; then
    CLEAN_GPUS=$(IFS=','; echo "${NUMA0[*]:0:4}")
elif [ ${#NUMA1[@]} -ge 4 ]; then
    CLEAN_GPUS=$(IFS=','; echo "${NUMA1[*]:0:4}")
else
    echo "ERROR: Cannot get 4 GPUs from one NUMA node. Got: $ALL_GPUS"; exit 1
fi

echo "All assigned GPUs: $ALL_GPUS"
echo "Using GPUs (single NUMA): $CLEAN_GPUS"

IFS=',' read -ra GPUS <<< "$CLEAN_GPUS"

source "$VENV_PATH/bin/activate"

# 3. Define Parameter Spaces
BENCHMARKS=("lu_mustard" "cholesky_mustard")
FREQS=(2040 1815 1740 1665 1590 1515 1440 1365 1290 1215 1140 1065 990 915 840 795 765 690 615 540 465 390 315 240)
TILE_SIZES=(16)
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
            TEMP_CSV_BASE="${ENERGY_DIR}/${BENCH}_${FREQ}_${TILE}_gpu"

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
            "$EXECUTABLE_PATH" -n="$N_SIZE" -t="$TILE" --static-multigpu -r="$RUNS" --output="${OUTPUT_DIR}/${BENCH}_${FREQ}" --measure=wait_start_ts,start_ts,end_ts

            # --- Step D: Stop Monitors & Reset ---
            for pid in "${MONITOR_PIDS[@]}"; do
                kill "$pid" 2>/dev/null
                (sleep 0.5 && kill -9 "$pid" 2>/dev/null) &
            done

            for i in "${!GPUS[@]}"; do
                sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "${GPUS[$i]}" --reset-clocks
            done
            sleep 2

            echo "END_EXPERIMENT: ${EXPERIMENT_ID}"
            echo "----------------------------------------------------"

        done
    done
done

deactivate