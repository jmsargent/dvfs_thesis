#!/bin/bash
#SBATCH --job-name=clock_test
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:2

# 1. Setup IDs from Slurm
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra ADDR <<< "$CLEAN_GPUS"

# 2. Unlock the permissions for your user session
# This is the "Admin" way to allow clock changes without sudo
nvidia-smi -i "$CLEAN_GPUS" -acp UNRESTRICTED

# 3. Define our test values
VAL_0=2025
VAL_1=2010

echo "Setting GPU ${ADDR[0]} -> ${VAL_0}MHz, GPU ${ADDR[1]} -> ${VAL_1}MHz"

# 4. Set the clocks (use min,max syntax to force a hard lock)
nvidia-smi -i "${ADDR[0]}" -lgc "$VAL_0","$VAL_0"
nvidia-smi -i "${ADDR[1]}" -lgc "$VAL_1","$VAL_1"

echo "----------------------------------------------------"
echo "Step 2: Verification"
echo "----------------------------------------------------"

# 5. Verify (clocks.max_gr is the 'Set' value for L4/Ada)
nvidia-smi -i "$CLEAN_GPUS" --query-gpu=index,clocks.gr,clocks.max_gr --format=csv

# 6. Reset for the next user
echo "----------------------------------------------------"
echo "Step 3: Resetting"
nvidia-smi -i "$CLEAN_GPUS" -rgc