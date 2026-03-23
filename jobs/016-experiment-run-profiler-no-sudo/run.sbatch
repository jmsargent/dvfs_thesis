# Define the base path on the host using the submission directory
PROFILER_OUT_BASE="${SLURM_SUBMIT_DIR}/benchmark_out"

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
    # Identify Ranks (evaluated inside the container)
    RANK=\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:-0}}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK:-\${MPI_LOCALRANKID:-0}}

    # The verified NVIDIA Python binary path
    PYTHON_BIN=\"/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/profilers/Nsight_Systems/target-linux-x64/python/bin/python\"

    # Run Profiler
    # PROFILER_OUT_BASE is expanded by host; RANK is expanded by container
    \$PYTHON_BIN \"$PROFILER_PATH\" --monitor \
    --gpu-id \"\$LOCAL_RANK\" \
    --interval 0.01 \
    --output \"${PROFILER_OUT_BASE}_\${RANK}.csv\" &

    PROFILER_PID=\$!
    sleep 1

    # Run Benchmark
    \"$EXECUTABLE_PATH\" -n=2400 -t=2 --tiled -r=1 --verbose

    # Cleanup
    kill \$PROFILER_PID
"