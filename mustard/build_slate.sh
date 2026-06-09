#!/bin/bash
# Build blaspp, lapackpp, and SLATE inside the container, install to ~/.local/slate.
# Run once before build_mustard.sh.
set -e

CONTAINER="/data/users/sargent/dvfs_thesis/containers/container.sif"
LOCAL_CMAKE="/data/users/sargent/mustard/cmake-3.28.1-linux-x86_64/bin/cmake"
PREFIX="$HOME/.local/slate"
SRC="$HOME/.local/src"

CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6"
BLAS_LIB_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/lib"
MPI_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi"

clone_if_missing() {
    local dir="$1" url="$2"
    if [ ! -d "$dir" ]; then
        apptainer exec "$CONTAINER" git clone --depth 1 "$url" "$dir"
    fi
}

clone_if_missing "$SRC/blaspp"   https://github.com/icl-utk-edu/blaspp.git
clone_if_missing "$SRC/lapackpp" https://github.com/icl-utk-edu/lapackpp.git
clone_if_missing "$SRC/slate"    https://github.com/icl-utk-edu/slate.git

cmake_build() {
    local src="$1"; shift
    local extra_args="$*"
    apptainer exec --nv "$CONTAINER" bash -c "
        rm -rf $src/build && mkdir -p $src/build && cd $src/build
        LD_LIBRARY_PATH=$BLAS_LIB_DIR:\$LD_LIBRARY_PATH \
        $LOCAL_CMAKE .. \
            -DCMAKE_INSTALL_PREFIX=$PREFIX \
            -DCMAKE_PREFIX_PATH=$PREFIX \
            -DCMAKE_CXX_COMPILER=g++ \
            -DCMAKE_CUDA_COMPILER=$CUDA_DIR/bin/nvcc \
            -DCMAKE_CUDA_ARCHITECTURES=89 \
            -DCMAKE_BUILD_TYPE=Release \
            -DBLAS_LIBRARIES=$BLAS_LIB_DIR/libblas.so \
            -DLAPACK_LIBRARIES=$BLAS_LIB_DIR/liblapack.so \
            $extra_args
        make -j\$(nproc) && make install
    "
}

echo "=== Building blaspp ==="
cmake_build "$SRC/blaspp" \
    -Dgpu_backend=cuda \
    -Dblas_fortran=add_

echo "=== Building lapackpp ==="
cmake_build "$SRC/lapackpp"

echo "=== Building SLATE ==="
cmake_build "$SRC/slate" \
    -Dgpu_backend=cuda \
    -DMPI_HOME=$MPI_HOME \
    "-DSCALAPACK_LIBRARIES=$HOME/.local/ompi/lib/libscalapack.so"

echo "SLATE installed to $PREFIX"
echo "Pass -Dslate_DIR=$PREFIX/lib/cmake/slate to CMake"
