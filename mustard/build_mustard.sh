#!/bin/bash

LOCAL_CMAKE="/data/users/sargent/mustard/cmake-3.28.1-linux-x86_64/bin/cmake"
CONTAINER="/data/users/sargent/dvfs_thesis/containers/container.sif"
INCLUDE_DIR="$HOME/.local/include"

# ── Sync headers from container ──────────────────────────────────────────────
echo "Syncing headers from container..."
mkdir -p "$INCLUDE_DIR"

apptainer exec "$CONTAINER" \
    cp -r /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/include/. \
    "$INCLUDE_DIR/cuda"

apptainer exec "$CONTAINER" \
    cp -r /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/include/. \
    "$INCLUDE_DIR/nvshmem"

apptainer exec "$CONTAINER" \
    cp -r /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include/. \
    "$INCLUDE_DIR/ompi"

apptainer exec "$CONTAINER" \
    cp -r /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/include/. \
    "$INCLUDE_DIR/cuda"

apptainer exec "$CONTAINER" \
    cp -r /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/targets/x86_64-linux/include/. \
    "$INCLUDE_DIR/cuda"

# ── Build ─────────────────────────────────────────────────────────────────────
apptainer exec --nv "$CONTAINER" bash -c "
    rm -rf build && mkdir -p build

    sed -i 's|nvshmem::nvshmem|-lnvshmem_host -lnvshmem_device|g' CMakeLists.txt

    cd build

    $LOCAL_CMAKE .. \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DMUSTARD_CUDA_ARCHITECTURES=89 \
        -DMUSTARD_BUILD_BASELINES=ON \
        -DCMAKE_EXE_LINKER_FLAGS=\"-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib\" \
        -DCMAKE_CUDA_FLAGS=\"-I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib -lnvshmem_host -lnvshmem_device -lcuda\"

    make -j\$(nproc)
"

# ── Patch .rsp and compile_commands.json to point to host include paths ───────
echo "Patching for IntelliSense..."

PATCH_ARGS=(
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/targets/x86_64-linux/include|$INCLUDE_DIR/cuda|g"
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/include|$INCLUDE_DIR/nvshmem|g"
    -e "s|/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/include|$INCLUDE_DIR/ompi|g"
)

find ~/dvfs_thesis/mustard/build -name "*.rsp" | while read rsp; do
    sed -i "${PATCH_ARGS[@]}" "$rsp"
done

sed -i "${PATCH_ARGS[@]}" ~/dvfs_thesis/mustard/build/compile_commands.json

# ── Inline .rsp files and patch for CLion IntelliSense ───────────────────────
echo "Inlining .rsp flags and patching compiler for CLion..."

python3 << 'PYEOF'
import json, os, re

build_dir = "/data/users/sargent/dvfs_thesis/mustard/build/mustard"
db_path = "/data/users/sargent/dvfs_thesis/mustard/build/compile_commands.json"

with open(db_path) as f:
    db = json.load(f)

for entry in db:
    cmd = entry["command"]

    # Inline .rsp files
    match = re.search(r'--options-file\s+(\S+)', cmd)
    if match:
        rsp_rel = match.group(1)
        rsp_abs = os.path.join(build_dir, rsp_rel)
        if os.path.exists(rsp_abs):
            with open(rsp_abs) as f:
                rsp_content = f.read().strip()
            cmd = cmd.replace(match.group(0), rsp_content)

    # Replace nvcc with g++ for IntelliSense
    cmd = cmd.replace(
        '/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6/bin/nvcc',
        '/usr/bin/g++'
    )

    # Remove nvcc-specific flags that g++ doesn't understand
    cmd = re.sub(r'-forward-unknown-to-host-compiler\s*', '', cmd)
    cmd = re.sub(r'--expt-extended-lambda\s*', '', cmd)
    cmd = re.sub(r'--expt-relaxed-constexpr\s*', '', cmd)
    cmd = re.sub(r'-use_fast_math\s*', '', cmd)
    cmd = re.sub(r'"--generate-code=\S+"\s*', '', cmd)
    cmd = re.sub(r'-rdc=true\s*', '', cmd)
    cmd = re.sub(r'-x cu\s*', '', cmd)
    cmd = re.sub(r'-Wno-deprecated-declarations\s*', '', cmd)

    entry["command"] = cmd

with open(db_path, "w") as f:
    json.dump(db, f, indent=2)

print("Done - compile_commands.json patched for CLion IntelliSense")
PYEOF

# ── Copy compile_commands.json to project root for CLion ─────────────────────
cp ~/dvfs_thesis/mustard/build/compile_commands.json ~/dvfs_thesis/
echo "Copied compile_commands.json to ~/dvfs_thesis/"