#!/bin/bash
#SBATCH --job-name=gpu_energy_study
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:2
#SBATCH --output=gpu_mon_%j.out

nvidia-smi -q -d SUPPORTED_CLOCKS | grep -E "Graphics|Memory" | head -n 20