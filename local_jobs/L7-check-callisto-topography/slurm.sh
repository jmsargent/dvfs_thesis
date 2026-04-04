#!/bin/bash
#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=1
#SBATCH --output=local_jobs/L7-check-callisto-topography/result_%j.out
#SBATCH --nodelist=callisto

CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"

echo "=== Node: $(hostname) ==="
echo "=== Date: $(date) ==="
echo ""

echo "=== GPU Topology ==="
nvidia-smi topo -m
echo ""

echo "=== PCIe Peer Access Matrix ==="
nvidia-smi topo -p2p r
echo ""

echo "=== GPU Info ==="
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv
echo ""

echo "=== Kernel / OS ==="
uname -a
echo ""

echo "=== Container: CUDA / NVSHMEM versions ==="
apptainer exec --nv "$CONTAINER_PATH" bash -c "
    nvcc --version 2>/dev/null || echo 'nvcc not found'
    echo ''
    ldconfig -p | grep nvshmem || echo 'nvshmem not in ldconfig'
    ls /usr/local/lib/libnvshmem* 2>/dev/null || echo 'no libnvshmem in /usr/local/lib'
"
echo ""

echo "=== Container: MPI peer connectivity test (no NVSHMEM) ==="
apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 4 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -mca pml ob1 \
    -mca btl self,vader \
    bash -c 'echo "rank $OMPI_COMM_WORLD_RANK on $(hostname)"'