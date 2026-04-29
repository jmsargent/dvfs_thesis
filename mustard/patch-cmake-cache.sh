#!/bin/bash
# Patch CMake cache to replace container paths with host paths
CACHE="$HOME/dvfs_thesis/mustard/cmake-build-debug/CMakeCache.txt"
COMPILER_CMAKE="$HOME/dvfs_thesis/mustard/cmake-build-debug/CMakeFiles/3.28.1/CMakeCUDACompiler.cmake"

PATCH_ARGS=(
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/include|/data/users/sargent/.local/include/cuda|g"
    -e "s|CUDA_nvml_LIBRARY:FILEPATH=CUDA_nvml_LIBRARY-NOTFOUND|CUDA_nvml_LIBRARY:FILEPATH=/data/users/sargent/.local/lib/libnvml.so|"
)

if [ -f "$CACHE" ]; then
    sed -i "${PATCH_ARGS[@]}" "$CACHE"
    echo "Patched CMakeCache.txt"
fi

if [ -f "$COMPILER_CMAKE" ]; then
    sed -i "${PATCH_ARGS[@]}" "$COMPILER_CMAKE"
    echo "Patched CMakeCUDACompiler.cmake"
fi

/data/users/sargent/dvfs_thesis/patch_cmake_cache.sh