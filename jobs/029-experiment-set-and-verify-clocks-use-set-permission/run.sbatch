#!/bin/bash
#SBATCH --job-name=clock_test
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

# 1. Setup IDs from Slurm
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
# Convert comma-separated string to array: (2 3)
IFS=',' read -ra ADDR <<< "$CLEAN_GPUS"

sudo nvidia-smi -i "$CLEAN_GPUS" -pm 1

# 2. Save all available clocks
mapfile -t ALL_CLOCKS < <(nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Graphics" | awk '{print $3}' | sort -unr)

# 3. Define our test values
VAL_GPU0=2025
VAL_GPU1=2010

echo "Using Physical GPUs: ${ADDR[0]} and ${ADDR[1]}"

# 4. Set the clocks using the mapped Slurm IDs
# Format: nvidia-smi -i <ID> -lgc <min_core,max_core>
nvidia-smi -i "${ADDR[0]}" -lgc "$VAL_GPU0","$VAL_GPU0"
nvidia-smi -i "${ADDR[1]}" -lgc "$VAL_GPU1","$VAL_GPU1"

echo "----------------------------------------------------"
echo "Step 2: Verification of 'Requested' vs 'Actual' Clocks"
echo "----------------------------------------------------"

# Verify specifically for the assigned IDs
nvidia-smi -i "$CLEAN_GPUS" --query-gpu=index,clocks.applications_gr,clocks.gr --format=csv

# 5. Reset clocks to default (Max Boost/Auto)
echo "----------------------------------------------------"
echo "Step 3: Resetting to Default (Max Boost)"
nvidia-smi -i "$CLEAN_GPUS" -rgc