#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:1

# Get the list of supported Graphics clocks for the primary Memory state
nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Graphics" | head -n 20