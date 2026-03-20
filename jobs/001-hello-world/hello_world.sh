#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=results/result_%j.out

echo "hello, world!"