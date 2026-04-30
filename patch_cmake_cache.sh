#!/bin/bash
BUILD_DIR="$HOME/dvfs_thesis/mustard/cmake-build-debug"

PATCH_ARGS=(
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/include|/data/users/sargent/.local/include/cuda|g"
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux|/data/users/sargent/.local/cuda-math|g"
    -e "s|CUDA_cublas_LIBRARY:FILEPATH=CUDA_cublas_LIBRARY-NOTFOUND|CUDA_cublas_LIBRARY:FILEPATH=/data/users/sargent/.local/cuda-math/lib/libcublas.so|"
    -e "s|CUDA_cublasLt_LIBRARY:FILEPATH=CUDA_cublasLt_LIBRARY-NOTFOUND|CUDA_cublasLt_LIBRARY:FILEPATH=/data/users/sargent/.local/cuda-math/lib/libcublasLt.so|"
    -e "s|CUDA_cusolver_LIBRARY:FILEPATH=CUDA_cusolver_LIBRARY-NOTFOUND|CUDA_cusolver_LIBRARY:FILEPATH=/data/users/sargent/.local/cuda-math/lib/libcusolver.so|"
    -e "s|CUDA_curand_LIBRARY:FILEPATH=CUDA_curand_LIBRARY-NOTFOUND|CUDA_curand_LIBRARY:FILEPATH=/data/users/sargent/.local/cuda-math/lib/libcurand.so|"
    -e "s|CUDA_nvml_LIBRARY:FILEPATH=CUDA_nvml_LIBRARY-NOTFOUND|CUDA_nvml_LIBRARY:FILEPATH=/data/users/sargent/.local/lib/libnvml.so|"
)

for f in \
    "$BUILD_DIR/CMakeCache.txt" \
    "$BUILD_DIR/CMakeFiles/3.28.1/CMakeCUDACompiler.cmake"; do
    if [ -f "$f" ]; then
        sed -i "${PATCH_ARGS[@]}" "$f"
        echo "Patched $f"
    fi
done
