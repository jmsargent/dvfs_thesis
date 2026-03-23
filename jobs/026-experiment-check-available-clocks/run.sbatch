#!/bin/bash
#SBATCH --job-name=test_run_benchmark
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:1

nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Graphics" | awk '{print $3}' | sort -unr