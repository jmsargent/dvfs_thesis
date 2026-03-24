#!/bin/bash
#SBATCH --job-name=gpu_energy_study
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:2
#SBATCH --output=gpu_mon_%j.out

# 1. Parse Slurm GPU IDs
GPU_LIST=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
CLEAN_GPUS=$(echo $GPU_LIST | tr -cd '0-9,')
IFS=',' read -ra ADDR <<< "$CLEAN_GPUS"

# 2. Unlock permissions (Still requires sudo or Persistence Mode + Admin level)
# If this fails on your cluster, contact your sysadmin to enable persistence mode.
nvidia-smi -i "$CLEAN_GPUS" -acp UNRESTRICTED

# 3. Frequency targets (L4 specific examples)
# Note: Energymonitor.py defaults memory to 6251 if not specified
VAL_GPU0=2025
VAL_GPU1=2010

echo "Applying frequencies via Energymonitor.py..."

# 4. Set frequencies for each GPU
# We use --set-freq without --monitor to just perform the clock lock
python3 Energymonitor.py --gpu-id ${ADDR[0]} --set-freq --gpu-freq $VAL_GPU0
python3 Energymonitor.py --gpu-id ${ADDR[1]} --set-freq --gpu-freq $VAL_GPU1

echo "----------------------------------------------------"
echo "Step 2: Monitoring during Workload"
echo "----------------------------------------------------"

# 5. Background monitoring on GPU 0 while running a workload
# Monitoring for 60 seconds at 100ms intervals
python3 Energymonitor.py --gpu-id ${ADDR[0]} --monitor --duration 60 --output stats_gpu0.csv &
MON_PID=$!

# Execute your actual HPC workload here
# e.g., ./my_cuda_app

wait $MON_PID

echo "----------------------------------------------------"
echo "Step 3: Resetting and Summary"
# 6. Reset clocks using your script's reset logic
python3 Energymonitor.py --gpu-id ${ADDR[0]} --reset-clocks
python3 Energymonitor.py --gpu-id ${ADDR[1]} --reset-clocks