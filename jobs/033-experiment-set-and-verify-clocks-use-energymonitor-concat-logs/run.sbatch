#!/bin/bash
#SBATCH --job-name=gpu_energy_study
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:2
#SBATCH --output=gpu_mon_%j.out

# 1. Paths (Updated from verified script)
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
OTHER_USER="chjing"
PROJECT_DIR="/data/users/${OTHER_USER}/dvfs_thesis"
PROFILER_PATH="${PROJECT_DIR}/chens_code/EnergyMonitor.py"
OUTPUT_FILE="stats_gpu0.csv"

# 2. Setup IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra ADDR <<< "$CLEAN_GPUS"

# 3. Environment Setup
source "$VENV_PATH/bin/activate"

# 4. Frequency targets
VAL_GPU0=2025
VAL_GPU1=2010

echo "Applying frequencies via EnergyMonitor.py..."

# 5. Set frequencies for each GPU
python3 "$PROFILER_PATH" --gpu-id ${ADDR[0]} --set-freq --gpu-freq $VAL_GPU0
python3 "$PROFILER_PATH" --gpu-id ${ADDR[1]} --set-freq --gpu-freq $VAL_GPU1

echo "----------------------------------------------------"
echo "Step 2: Monitoring"
echo "----------------------------------------------------"

# 6. Background monitoring on GPU 0
python3 "$PROFILER_PATH" --gpu-id ${ADDR[0]} --monitor --duration 2 --output "$OUTPUT_FILE" &
MON_PID=$!

# Wait for the 2-second monitor to complete
wait $MON_PID

echo "----------------------------------------------------"
echo "Step 3: Resetting Clocks"

# 7. Reset clocks using script's reset logic
python3 "$PROFILER_PATH" --gpu-id ${ADDR[0]} --reset-clocks
python3 "$PROFILER_PATH" --gpu-id ${ADDR[1]} --reset-clocks

deactivate

echo "----------------------------------------------------"
echo "Step 4: Consolidating Results"
echo "----------------------------------------------------"

# 8. Dump CSV data to Slurm log and cleanup
if [ -f "$OUTPUT_FILE" ]; then
    echo "--- BEGIN PROFILER DATA GPU ${ADDR[0]} ---"
    cat "$OUTPUT_FILE"
    rm "$OUTPUT_FILE"
    echo "--- END PROFILER DATA GPU ${ADDR[0]} ---"
else
    echo "ERROR: $OUTPUT_FILE not found."
fi