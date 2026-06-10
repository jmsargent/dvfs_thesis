#!/bin/bash
#SBATCH --job-name=baselines_comparison
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:L4:4
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --output=slurm_job_history/result_%j.out
#SBATCH --nodelist=ganymede,europa
#SBATCH --exclusive

source /data/users/sargent/dvfs_thesis/benchscripts/library.sh

BUILD="$BIN_DIR"
CLEAN_GPUS=$(select_gpus 4)

_bench_idx=0
bench_tag() {
    _bench_idx=$((_bench_idx + 1))
    printf 'BENCH idx=%d solver=%s N=%s NB=%s runs=%s\n' \
        "$_bench_idx" "$1" "$2" "$3" "$4"
}

run_bench(){
    local N="$1"
    local TILE_SIZE="$2"
    local RUNS="$3"
    local T=$((N / TILE_SIZE))
    local NB=$TILE_SIZE

    echo "======== N=$N tile_size=$TILE_SIZE tiles=$T runs=$RUNS ========"

    bench_tag mustard_lu     "$N" "$NB" "$RUNS"
    run --np 4 --gpus "$CLEAN_GPUS" --symmetric-size 14G -- \
       "$BUILD/lu_mustard" -n="$N" -t="$T" --panel -r="$RUNS" --scaler=none

    bench_tag mustard_chol   "$N" "$NB" "$RUNS"
    run --np 4 --gpus "$CLEAN_GPUS" --symmetric-size 14G -- \
       "$BUILD/cholesky_mustard" -n="$N" -t="$T" --panel -r="$RUNS" --scaler=none

    bench_tag slate_lu       "$N" "$NB" "$RUNS"
    apptainer_run mpirun -np 4 \
        --bind-to core \
        -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
        -x OMP_NUM_THREADS=1 \
        -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
        -mca plm ^slurm -mca coll_hcoll_enable 0 \
        -- "$BUILD/lu_slate" -N="$N" -B="$NB" -r="$RUNS"

    bench_tag slate_chol     "$N" "$NB" "$RUNS"
    apptainer_run mpirun -np 4 \
        --bind-to core \
        -x CUDA_VISIBLE_DEVICES="$CLEAN_GPUS" \
        -x OMP_NUM_THREADS=1 \
        -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
        -mca plm ^slurm -mca coll_hcoll_enable 0 \
        -- "$BUILD/chol_slate" -N="$N" -B="$NB" -r="$RUNS"

    bench_tag starpu_lu      "$N" "$NB" "$RUNS"
    apptainer_run env STARPU_NCUDA=4 STARPU_NCPU=0 \
        "$BUILD/starpu_lu_example" -size "$N" -nblocks "$T"

    bench_tag starpu_chol    "$N" "$NB" "$RUNS"
    for _r in $(seq 1 "$RUNS"); do
        apptainer_run env STARPU_NCUDA=4 STARPU_NCPU=0 \
            "$BUILD/starpu_cholesky_tile_tag" -size "$N" -nblocks "$T"
    done
}

SIZES=(6000 8000 12000 18000 24000)
TILE_SIZE=1000
echo "######## SWEEP 1: Matrix size sweep with TILE_SIZE=1000 ########"
for N in "${SIZES[@]}"; do
    run_bench "$N" "$TILE_SIZE" 5
done

TILE_SIZES=(100 200 300 400 500 600 750 1000 1500 2000 3000)
echo "######## SWEEP 2: tile size sweep N=12000 ########"
for NB in "${TILE_SIZES[@]}"; do
    run_bench 12000 "$NB" 5
done
