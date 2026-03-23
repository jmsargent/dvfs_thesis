# 1. Prepare host-side variables
PROFILER_OUT_BASE="${SLURM_SUBMIT_DIR}/benchmark_out"
NVIDIA_PY="/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/profilers/Nsight_Systems/target-linux-x64/python/bin/python"

# 2. Run the container
apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" mpirun -np 2 \
--bind-to core \
-x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
-x NVSHMEM_BOOTSTRAP=MPI \
-x NVSHMEM_REMOTE_TRANSPORT=none \
-x UCX_TLS=sm,cuda_copy,cuda_ipc,self \
-x UCX_NET_DEVICES=all \
-mca pml ob1 \
-mca btl self,vader \
-mca coll_hcoll_enable 0 \
/bin/bash -c "
    # Internal variables (escaped with backslash)
    RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

    # Start Profiler using the absolute path to NVIDIA's Python
    # We use the variable defined on the host ($NVIDIA_PY)
    \"$NVIDIA_PY\" \"$PROFILER_PATH\" --monitor \
    --gpu-id \"\$LOCAL_RANK\" \
    --interval 0.01 \
    --output \"${PROFILER_OUT_BASE}_\${RANK}.csv\" &

    # Capture PID and wait for initialization
    P_PID=\$!
    sleep 2

    # Execute Benchmark
    if [ -f \"$EXECUTABLE_PATH\" ]; then
        \"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose
    else
        echo \"ERROR: Executable not found at $EXECUTABLE_PATH\"
        exit 1
    fi

    # Cleanup
    kill \$P_PID
"