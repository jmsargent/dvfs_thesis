#!/bin/bash
ARGS=()
for arg in "$@"; do
    case "$arg" in
        -ccbin=*) ;;
        *) ARGS+=("$arg") ;;
    esac
done

/usr/bin/apptainer exec --nv \
  -B /tmp \
  -B /data/users/sargent \
  -B /data/users/sargent/opt/nvidia:/data/users/sargent/opt/nvidia \
  /data/users/sargent/dvfs_thesis/containers/container.sif \
  /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/bin/nvcc -ccbin=g++ "${ARGS[@]}"
