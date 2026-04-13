#!/bin/bash
#SBATCH --job-name=query_l4_clocks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:1
#SBATCH --nodelist=ganymede,europa

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0

echo ""
echo "=== Supported Graphics (SM) Clock Frequencies (MHz) ==="
nvidia-smi --query-supported-clocks=gr --format=csv,noheader -i 0

echo ""
echo ""
echo "=== Supported Memory Clock Frequencies (MHz) ==="
nvidia-smi --query-supported-clocks=mem --format=csv,noheader -i 0
