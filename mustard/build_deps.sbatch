#!/bin/bash
#SBATCH --job-name=build_deps
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --output=build_deps_%j.out
#SBATCH --nodelist=io

cd /data/users/sargent/dvfs_thesis/mustard

# echo "=== Building StarPU ==="
# bash /data/users/sargent/dvfs_thesis/mustard/build_starpu.sh

echo "=== Building SLATE ==="
bash /data/users/sargent/dvfs_thesis/mustard/build_slate.sh

echo "=== Done ==="
