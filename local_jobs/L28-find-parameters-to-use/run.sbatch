#!/bin/bash
#SBATCH --job-name=find-params
#SBATCH --output=local_jobs/L28-find-parameters-to-use/result_%j.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:L4:6
#SBATCH --nodelist=europa,ganymede,io
#SBATCH --nodes=1
#SBATCH --partition=long



source "/data/users/sargent/dvfs_thesis/benchscripts/library.sh"
source "$VENV_PATH/bin/activate"

CLEAN_GPUS=$(select_gpus 4)
SCRIPT_DIR="/data/users/sargent/dvfs_thesis/local_jobs/L28-find-parameters-to-use"
RUNS=5
MAX_SECS=1200  # 20 minutes

echo "got clean gpus: $CLEAN_GPUS"

LAST_ELAPSED=0
run_bench() {
    local bench=$1 n=$2 t=$3 streams=$4
    local dir="${SCRIPT_DIR}/results/${bench}/n${n}_t${t}_s${streams}"
    mkdir -p "$dir"
    echo "--- $bench n=$n t=$t streams=$streams ---"
    local t0
    t0=$(date +%s)
    run --np "$SLURM_NTASKS" --gpus "$CLEAN_GPUS" --symmetric-size 14G \
        "$BENCHMARK" \
            --name="$bench" \
            -n="$n" \
            -t="$t" \
            --panel \
            -r="$RUNS" \
            --streams="$streams"
    LAST_ELAPSED=$(( $(date +%s) - t0 ))
    echo "wall time: ${LAST_ELAPSED}s"
}

for BENCH in cholesky lu; do
    echo ""
    echo "=========================================="
    echo "BENCHMARK: $BENCH"
    echo "=========================================="

    # Phase 1: largest N where 5 runs fit in 20 minutes (T=16, streams=1)
    echo ""
    echo "--- Phase 1: N sweep (T=16, streams=1) ---"
    BEST_N=20000
    for N in 20000 24000 28000 32000 36000 40000 44000 48000; do
        run_bench "$BENCH" "$N" 16 1
        if (( LAST_ELAPSED > MAX_SECS )); then
            echo "N=$N exceeded 20 min, keeping BEST_N=$BEST_N"
            break
        fi
        BEST_N=$N
        echo "BEST_N=$BEST_N"
    done
    echo "Phase 1 done: BEST_N=$BEST_N"

    # Phase 2: fastest T at BEST_N (streams=1)
    echo ""
    echo "--- Phase 2: T sweep (N=$BEST_N, streams=1) ---"
    BEST_T=16
    BEST_T_ELAPSED=999999
    for T in 8 16 20 24 32 40 48 64; do
        if (( BEST_N % T != 0 )); then
            echo "T=$T skipped (${BEST_N} not divisible)"
            continue
        fi
        run_bench "$BENCH" "$BEST_N" "$T" 1
        if (( LAST_ELAPSED < BEST_T_ELAPSED )); then
            BEST_T_ELAPSED=$LAST_ELAPSED
            BEST_T=$T
        fi
        echo "BEST_T so far: $BEST_T (${BEST_T_ELAPSED}s)"
    done
    echo "Phase 2 done: BEST_T=$BEST_T"

    # Phase 3: fastest streams at BEST_N, BEST_T
    echo ""
    echo "--- Phase 3: streams sweep (N=$BEST_N, T=$BEST_T) ---"
    BEST_S=1
    BEST_S_ELAPSED=999999
    for S in 1 2 4 8 16 32; do
        run_bench "$BENCH" "$BEST_N" "$BEST_T" "$S"
        if (( LAST_ELAPSED < BEST_S_ELAPSED )); then
            BEST_S_ELAPSED=$LAST_ELAPSED
            BEST_S=$S
        fi
        echo "BEST_S so far: $BEST_S (${BEST_S_ELAPSED}s)"
    done
    echo "Phase 3 done: BEST_STREAMS=$BEST_S"

    echo ""
    echo "=== $BENCH OPTIMAL: N=$BEST_N T=$BEST_T streams=$BEST_S ==="
done

deactivate
