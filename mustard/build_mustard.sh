#!/bin/bash

# Use your local modern CMake binary
LOCAL_CMAKE="/data/users/sargent/mustard/cmake-3.28.1-linux-x86_64/bin/cmake"

apptainer exec --nv /data/users/sargent/dvfs_thesis/containers/container.sif bash -c "
    # 1. Clean
    rm -rf build && mkdir -p build

    # 2. Patch (ensure we are in root)
    sed -i 's|nvshmem::nvshmem|-lnvshmem_host -lnvshmem_device|g' CMakeLists.txt

    cd build

    # 3. Configure using Absolute Internal Paths
    # Note: I'm using the paths defined in your .def file
    $LOCAL_CMAKE .. \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DMUSTARD_CUDA_ARCHITECTURES=89 \
        -DCMAKE_EXE_LINKER_FLAGS=\"-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib\" \
        -DCMAKE_CUDA_FLAGS=\"-I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib -lnvshmem_host -lnvshmem_device -lcuda\"

    # 4. Build
    make -j\$(nproc)
"