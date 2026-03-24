#!/bin/bash

# --- Configuration --- #
# Set your target device ID
DEVICE_ID=0

APPLICATIONS=(
    # "./07-cholesky 32768 4096"
    "./07-cholesky 8192 2048"
)

#list of kernels to include
KERNEL_NAMES=("POTRF" "SYRK" "GEMM" "TRSM")
# Define launch counts per kernel
LAUNCH_COUNTS=(2 1 1 1)  # POTRF, SYRK, GEMM, TRSM

# Set the full list of metrics you want to collect
# Using a variable makes the main loop cleaner

# METRICS_LIST="smsp__cycles_active.sum"

# METRICS_LIST="smsp__cycles_active.sum,\
# smsp__thread_inst_executed_pipe_alu_pred_on.sum,\
# smsp__thread_inst_executed_pipe_fma_pred_on.sum,\
# smsp__thread_inst_executed_pipe_fp16_pred_on.sum,\
# smsp__thread_inst_executed_pipe_fp64_pred_on.sum,\
# smsp__thread_inst_executed_pipe_xu_pred_on.sum,\
# smsp__inst_executed_pipe_tensor_op_dmma.sum,\
# smsp__inst_executed_pipe_tensor_op_hmma.sum,\
# smsp__inst_executed_pipe_tensor_op_imma.sum,\
# l1tex__t_sector_hit_rate.pct,\
# dram__bytes.sum,\
# lts__t_bytes.sum,\
# gpu__time_duration.sum,\
# smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared.sum,\
# l1tex__data_pipe_lsu_wavefronts.sum"

METRICS_LIST="smsp__cycles_active.sum,\
smsp__thread_inst_executed_pipe_alu_pred_on.sum,\
smsp__thread_inst_executed_pipe_fma_pred_on.sum,\
smsp__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum,\
smsp__thread_inst_executed_pipe_fp64_pred_on.sum,\
smsp__thread_inst_executed_pipe_xu_pred_on.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_imma_cycles_active.sum,\
l1tex__t_sector_hit_rate.pct,\
dram__bytes.sum,\
lts__t_bytes.sum,\
gpu__time_duration.sum,\
smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__data_pipe_lsu_wavefronts.sum"

# Number of repetitions per benchmark
REPS=1


# --- Main Execution Loop --- #

# Output directory
OUT_DIR="L4_Cholesky_NCU_Profile"
mkdir -p "$OUT_DIR"

echo "Starting NCU profiling runs..."

# Path to your Conda environment
CONDA_ENV_PATH="/data/users/chjing/miniforge3/envs/cuda-env"

# Add Conda libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

# --- Profiling Loop --- #
for APP in "${APPLICATIONS[@]}"; do
	APP_BIN=$(echo $APP | awk '{print $1}')
    APP_ARGS=$(echo $APP | cut -d' ' -f2-)
    APP_NAME=$(basename "$APP_BIN")

    for ((i=1; i<=REPS; i++)); do
        echo "----------------------------------------------------"
        echo "Running $APP_NAME (run $i/$REPS)"

        # OUTPUT_FILE="$OUT_DIR/${APP_NAME}_run${i}.csv"

        # ncu --nvtx --nvtx-include "POTRF/" \
        #     --nvtx-include "SYRK/" \
        #     --nvtx-include "GEMM/" \
        #     --nvtx-include "TRSM/" \
        #     --target-processes all \
        #     --device $DEVICE_ID \
        #     --csv \
        #     --page details \
        #     --metrics "$METRICS_LIST" \
        #     $APP_BIN $APP_ARGS > "$OUTPUT_FILE" 

        ##--kernel-name regex:"gemm|getrf|trsm|syrk"  --kernel-name-base demangled \
        for i in "${!KERNEL_NAMES[@]}"; do
            KERNEL="${KERNEL_NAMES[$i]}"
            COUNT="${LAUNCH_COUNTS[$i]}"
            
            OUTFILE="$OUT_DIR/${KERNEL}.csv"
            echo "Profiling kernel $KERNEL -> $OUTFILE"

            export CUDASTF_SCHEDULE=random
            export CUDA_VISIBLE_DEVICES=2

            /data/users/chjing/miniforge3/envs/cuda-env/nsight-compute/2024.1.1/ncu --nvtx --nvtx-include "${KERNEL}/" \
                --target-processes all \
                --device $DEVICE_ID \
                --csv \
                --page details \
                --metrics "$METRICS_LIST" \
                --launch-count "${COUNT}" \
                --kill yes \
                $APP_BIN $APP_ARGS > "$OUTFILE"

            if [ $? -eq 0 ]; then
                echo "✅ Successfully generated: $OUTFILE"
            else
                echo "❌ Error running NCU for $APP_NAME (run $i)"
            fi
        done
    done
done

echo "----------------------------------------------------"
echo "All profiling runs completed."
