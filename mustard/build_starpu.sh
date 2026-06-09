#!/bin/bash
# Build StarPU 1.3.x inside the container, install to ~/.local/starpu.
# Run once before build_mustard.sh.
set -e

CONTAINER="/data/users/sargent/dvfs_thesis/containers/container.sif"
PREFIX="$HOME/.local/starpu"
SRC="$HOME/.local/src/starpu"

CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6"
CUDA_MATH_LIB="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/lib"
BLAS_LIB_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/lib"

TARBALL="starpu-1.3.11.tar.gz"
TARBALL_URL="https://files.inria.fr/starpu/starpu-1.3.11/$TARBALL"

if [ ! -d "$SRC" ]; then
    mkdir -p "$(dirname "$SRC")"
    wget -q --show-progress -O "/tmp/$TARBALL" "$TARBALL_URL"
    tar -xf "/tmp/$TARBALL" -C "$(dirname "$SRC")"
    mv "$(dirname "$SRC")/starpu-1.3.11" "$SRC"
    rm "/tmp/$TARBALL"
fi

apptainer exec --nv "$CONTAINER" bash -c "
    mkdir -p $SRC/_build && cd $SRC/_build

    CUDA_INC=$CUDA_DIR/targets/x86_64-linux/include
    CUDA_LIB=$CUDA_DIR/targets/x86_64-linux/lib

    $SRC/configure \
        --prefix=$PREFIX \
        --enable-cuda \
        --disable-opencl \
        --disable-mpi \
        --without-hwloc \
        NVCC=$CUDA_DIR/bin/nvcc \
        CPPFLAGS=\"-I\$CUDA_INC\" \
        LDFLAGS=\"-L\$CUDA_LIB -L$CUDA_MATH_LIB -L$BLAS_LIB_DIR\" \
        LIBS=\"-lblas -llapack\"

    make -j\$(nproc)
    make install
"

echo "StarPU installed to $PREFIX"
echo "Add to PKG_CONFIG_PATH: $PREFIX/lib/pkgconfig"
