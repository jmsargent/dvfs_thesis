#!/bin/bash

# ==============================================================================
# 1. GLOBAL CONFIGURATION - CHANGE ONLY THIS SECTION
# ==============================================================================
# Options: random, round_robin, heft, energy_aware
SCHED_TYPE="heft"

# Benchmark Parameters
MATRIX_SIZES=(32768)
BLOCK_SIZES=(2048)
RUNS=1

# Environment Setup
export CUDA_VISIBLE_DEVICES=0
CONDA_ENV_PATH="/data/users/chjing/miniforge3/envs/cuda-env"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

#L4
SUPPORTED_MEM_FREQUENCIES=("6251")

GPU_FREQ=("2040" "2025" "2010" "1995" "1980" "1965" "1950" "1935" "1920" "1905" "1890" "1875" "1860" "1845" "1830" "1815" "1800" "1785" "1770" "1755" "1740" "1725" "1710" "1695" "1680" "1665" "1650" "1635" "1620" "1605" "1590" "1575" "1560" "1545" "1530" "1515" "1500" "1485" "1470" "1455" "1440" "1425" "1410" "1395" "1380" "1365" "1350" "1335" "1320" "1305" "1290" "1275" "1260" "1245" "1230" "1215" "1200" "1185" "1170" "1155" "1140" "1125" "1110" "1095" "1080" "1065" "1050" "1035" "1020" "1005" "990" "975" "960" "945" "930" "915" "900" "885" "870" "855" "840" "825" "810" "795" "780" "765" "750" "735" "720" "705" "690" "675" "660" "645" "630" "615" "600" "585" "570" "555" "540" "525" "510" "495" "480" "465" "450" "435" "420" "405" "390" "375" "360" "345" "330" "315" "300" "285" "270" "255" "240" "225" "210")

#L40S
# SUPPORTED_MEM_FREQUENCIES=("9001")

# GPU_FREQ=("2520" "2505" "2490" "2475" "2460" "2445" "2430" "2415" "2400" "2385" "2370" "2355" "2340" "2325" "2310" "2295" "2280" "2265" "2250" "2235" "2220" "2205" "2190" "2175" "2160" "2145" "2130" "2115" "2100" "2085" "2070" "2055" "2040" "2025" "2010" "1995" "1980" "1965" "1950" "1935" "1920" "1905" "1890" "1875" "1860" "1845" "1830" "1815" "1800" "1785" "1770" "1755" "1740" "1725" "1710" "1695" "1680" "1665" "1650" "1635" "1620" "1605" "1590" "1575" "1560" "1545" "1530" "1515" "1500" "1485" "1470" "1455" "1440" "1425" "1410" "1395" "1380" "1365" "1350" "1335" "1320" "1305" "1290" "1275" "1260" "1245" "1230" "1215" "1200" "1185" "1170" "1155" "1140" "1125" "1110" "1095" "1080" "1065" "1050" "1035" "1020" "1005" "990" "975" "960" "945" "930" "915" "900" "885" "870" "855" "840" "825" "810" "795" "780" "765" "750" "735" "720" "705" "690" "675" "660" "645" "630" "615" "600" "585" "570" "555" "540" "525" "510" "495" "480" "465" "450" "435" "420" "405" "390" "375" "360" "345" "330" "315" "300" "285" "270" "255" "240" "225" "210")

# ==============================================================================
# 2. DYNAMIC SCHEDULER SETUP
# ==============================================================================
OUTPUT_DIR="L40S_Cholesky_Calibration_Results"
mkdir -p "$OUTPUT_DIR"
RESULTS_SUMMARY="$OUTPUT_DIR/Calibration_summary_results.txt"

export CUDASTF_SCHEDULE=$SCHED_TYPE

case $SCHED_TYPE in
    "heft")
        export CUDASTF_CALIBRATION_FILE="calibration.csv"
        # export CUDASTF_TASK_STATISTICS="cholesky_heft_calibration.csv"
        ;;
    "energy_aware")
        export CUDASTF_ENERGY_STATISTICS="2048_energy_profile.csv"
        export CUDASTF_ENERGY_WEIGHT=1  # Energy exponent (m)
        export CUDASTF_TIME_WEIGHT=0    # Time exponent (n)
        export CUDASTF_IDLE_POWER_WATTS=15.9
        ;;
esac

# Additional CUDASTF Globals
export USER_ALLOC_POOLS_MEM_CAP=0.80
# export CUDASTF_DEBUG=1 # Uncomment for verbose scheduler output

# ==============================================================================
# 3. MONITORING SETUP
# ==============================================================================
MONITOR_SCRIPT="gpu_monitor.py"
MONITOR_GPUS=(0) 
GPU_ID=${MONITOR_GPUS[0]}
POWER_COLUMN_INDEX=3 
MONITOR_PIDS=()

# --- Cleanup Function ---
cleanup() {
    echo ""
    echo "--- Cleanup ---"
    if [ ${#MONITOR_PIDS[@]} -eq 0 ]; then
        echo "No active monitor processes to clean up."
        return
    fi
    
    # Send SIGTERM to all monitors
    for PID in "${MONITOR_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "Sending SIGTERM to monitor (PID: $PID)..."
            kill -TERM "$PID" 2>/dev/null || true
        fi
    done
    
    # Wait for graceful shutdown
    echo "Waiting for monitors to flush data and terminate..."
    for attempt in {1..10}; do
        all_dead=true
        for PID in "${MONITOR_PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                all_dead=false
                break
            fi
        done
        
        if $all_dead; then
            echo "All monitors terminated gracefully."
            break
        fi
        sleep 0.5
    done
    
    # Force kill any remaining processes
    for PID in "${MONITOR_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing monitor (PID: $PID)..."
            kill -KILL "$PID" 2>/dev/null || true
        fi
    done
    
    # Clear the array
    MONITOR_PIDS=()
    
    # Force filesystem sync to ensure files are written
    sync
    sleep 1
}

# Set trap to ensure monitor is killed on script exit or interruption
trap cleanup EXIT INT TERM

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================
echo "Starting: $SCHED_TYPE"
echo "SIZE,BLOCK_SIZE,RUN,GPU_ID,AVG_POWER_W,TOTAL_ENERGY_J,DURATION_S,SAMPLES_COUNT" > "$RESULTS_SUMMARY"

for MEM_FREQ in "${SUPPORTED_MEM_FREQUENCIES[@]}"; do
    for GPU_FREQ in "${GPU_FREQ[@]}"; do
        echo "Setting GPU Frequency: Mem=${MEM_FREQ} MHz, GPU=${GPU_FREQ} MHz"
        # Set frequency
        sudo -E $(which python3) "$MONITOR_SCRIPT" --gpu-id $GPU_ID --set-freq --gpu-freq "$GPU_FREQ" --mem-freq "$MEM_FREQ"
        
        if [ $? -ne 0 ]; then
            echo "  ✗ ERROR: Failed to set frequency. Skipping."
            continue
        fi
        
        sleep 2

        # Verify application clocks instead of current clocks
        APP_GPU_FREQ=$(nvidia-smi -i $GPU_ID --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits)
        APP_MEM_FREQ=$(nvidia-smi -i $GPU_ID --query-gpu=clocks.applications.memory --format=csv,noheader,nounits)

        if [ -n "$APP_GPU_FREQ" ] && [ -n "$APP_MEM_FREQ" ]; then
            echo "  Verified (application clocks): GPU ${APP_GPU_FREQ} MHz, Memory ${APP_MEM_FREQ} MHz"
        else
            echo "  ⚠️ Warning: Unable to verify application clocks."
        fi
        
        for SIZE in "${MATRIX_SIZES[@]}"; do
            for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                echo "===== SIZE=$SIZE BLOCK_SIZE=$BLOCK_SIZE Policy=$SCHED_TYPE ====="
                
                for ((i=1; i<=RUNS; i++)); do
                    echo "--- Run $i of $RUNS ---"
                    
                    # --- 4A. Start Monitors ---
                    MONITOR_PIDS=()
                    PRIMARY_LOG_FILE=""
                    
                    # for GID in "${MONITOR_GPUS[@]}"; do
                    LOG_FILE="$OUTPUT_DIR/power_${MEM_FREQ}_${GPU_FREQ}.csv"
                    if [ -z "$PRIMARY_LOG_FILE" ]; then PRIMARY_LOG_FILE="$LOG_FILE"; fi

                    echo "   Starting background power monitor for GPU $GPU_ID. Log: $LOG_FILE"
                    $(which python3) "$MONITOR_SCRIPT" --monitor --interval 0.01 --output "$LOG_FILE" --gpu-id $GPU_ID &
                    PID=$!
                    MONITOR_PIDS+=($PID)
                    echo "   Monitor started with PID: $PID"
                    # done

                    # Wait for the primary monitor (first in the array) to initialize
                    for k in {1..20}; do
                        if [ -f "$PRIMARY_LOG_FILE" ] && [ -s "$PRIMARY_LOG_FILE" ]; then
                            echo "   Primary monitor initialized successfully"
                            break
                        fi
                        sleep 0.5
                    done

                    # Verify all monitors started
                    for PID in "${MONITOR_PIDS[@]}"; do
                        if ! kill -0 $PID 2>/dev/null; then
                            echo "   ✗ ERROR: Power monitor with PID $PID failed to start. Exiting."
                            exit 1
                        fi
                    done

                    # --- 4B. Run Benchmark ---
                    BENCHMARK_BIN="./07-cholesky"
                    BENCHMARK_ARGS="$SIZE $BLOCK_SIZE"
                    TEMP_OUTPUT="$OUTPUT_DIR/cholesky_calibration_${MEM_FREQ}_${GPU_FREQ}.txt"

                    echo "   Executing $BENCHMARK_BIN..."
                    sudo -E LD_LIBRARY_PATH="$LD_LIBRARY_PATH" bash -c "$BENCHMARK_BIN $BENCHMARK_ARGS" > "$TEMP_OUTPUT" 2>&1
                    
                    # --- 4C. Stop Monitors ---
                    cleanup
                    sleep 5
                done
            done
        done
    done
done

# Disable the trap before final exit to prevent running cleanup twice
trap - EXIT
# Good practice to unset them after the run
unset CUDASTF_DOT_FILE
unset CUDASTF_SCHEDULE
unset USER_ALLOC_POOLS_MEM_CAP
unset CUDASTF_ENERGY_STATISTICS
unset CUDASTF_ENERGY_WEIGHT
unset CUDASTF_TIME_WEIGHT
unset CUDASTF_IDLE_POWER_WATTS

echo "================================================================"
echo "🏁 All runs completed for $SCHED_TYPE. Summary: $RESULTS_SUMMARY"