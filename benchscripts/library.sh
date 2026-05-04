
CONTAINER_PATH="/data/users/sargent/dvfs_thesis/containers/container.sif"
PROJECT_DIR="/data/users/sargent/dvfs_thesis"
BIN_DIR="${PROJECT_DIR}/mustard/build"
VENV_PATH="/data/users/sargent/dvfs_thesis/venv_hpc"
PROFILER_PATH="/data/users/chjing/dvfs_thesis/chens_code/EnergyMonitor.py"

NVSHMEM_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/12.6/nvshmem/lib
CONTAINER_LD=/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/nvshmem/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/comm_libs/nccl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/lib64:/.singularity.d/libs


# Selects gpus from single numa
select_gpus() {
    local ngpu="${1:-$SLURM_NTASKS}"
    local gpu_list all_gpus all_gpu_arr numa0=() numa1=()

    gpu_list=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
    all_gpus=$(echo "$gpu_list" | tr -cd '0-9,')
    IFS=',' read -ra all_gpu_arr <<< "$all_gpus"

    for gpu in "${all_gpu_arr[@]}"; do
        [ "$gpu" -lt "$ngpu" ] && numa0+=("$gpu") || numa1+=("$gpu")
    done

    if [ ${#numa0[@]} -ge "$ngpu" ]; then
        IFS=','; echo "${numa0[*]:0:$ngpu}"
    elif [ ${#numa1[@]} -ge "$ngpu" ]; then
        IFS=','; echo "${numa1[*]:0:$ngpu}"
    else
        echo "ERROR: Cannot get $ngpu GPUs from one NUMA node. Got: $all_gpus" >&2; return 1
    fi
}

apptainer_run() {
    APPTAINERENV_LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CONTAINER_LD}" \
    apptainer exec --nv --bind /dev/shm,/data "$CONTAINER_PATH" "$@"
}

run() {
    apptainer_run "${PROJECT_DIR}/benchscripts/mpirun.sh" "$@"
}

BENCHMARK="${PROJECT_DIR}/benchscripts/benchmark.sh"

benchmark() {
    "$BENCHMARK" "$@"
}

start_monitors() {
    local csv_base="$1" clean_gpus="$2"
    [[ "${VIRTUAL_ENV:-}" == "$VENV_PATH" ]] || { echo "ERROR: expected venv $VENV_PATH, got '${VIRTUAL_ENV:-none}'" >&2; return 1; }
    local i=0 gpu
    IFS=',' read -ra _gpus <<< "$clean_gpus"
    for gpu in "${_gpus[@]}"; do
        python3 "$PROFILER_PATH" --monitor \
            --gpu-id "$gpu" \
            --interval 0.01 \
            --output "${csv_base}_${i}.csv" >/dev/null 2>/dev/null &
        echo $!
        (( i++ ))
    done
}

stop_monitors() {
    local pid
    for pid in "$@"; do
        kill "$pid" 2>/dev/null
    done
    for pid in "$@"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 0.1
        done
    done
    for pid in "$@"; do
        kill -9 "$pid" 2>/dev/null
    done
}

set_gpu_freq() {
    local clean_gpus="$1" freq="$2"
    [[ "${VIRTUAL_ENV:-}" == "$VENV_PATH" ]] || { echo "ERROR: expected venv $VENV_PATH, got '${VIRTUAL_ENV:-none}'" >&2; return 1; }
    local gpu
    IFS=',' read -ra _gpus <<< "$clean_gpus"
    for gpu in "${_gpus[@]}"; do
        sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "$gpu" --set-freq --gpu-freq "$freq"
    done
}

reset_gpu_clocks() {
    local clean_gpus="$1"
    [[ "${VIRTUAL_ENV:-}" == "$VENV_PATH" ]] || { echo "ERROR: expected venv $VENV_PATH, got '${VIRTUAL_ENV:-none}'" >&2; return 1; }
    local gpu
    IFS=',' read -ra _gpus <<< "$clean_gpus"
    for gpu in "${_gpus[@]}"; do
        sudo -E "$(which python3)" "$PROFILER_PATH" --gpu-id "$gpu" --reset-clocks
    done
}
