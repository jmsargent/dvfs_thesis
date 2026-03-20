#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "hello, world!"