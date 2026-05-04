#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/library.sh"

ok=true

check_file() {
    local name="$1" path="$2"
    [[ -f "$path" ]] || { echo "MISSING file:      $name=$path"; ok=false; }
}

check_dir() {
    local name="$1" path="$2"
    [[ -d "$path" ]] || { echo "MISSING directory: $name=$path"; ok=false; }
}

check_dir  PROJECT_DIR      "$PROJECT_DIR"
check_dir  BIN_DIR          "$BIN_DIR"
check_dir  NVSHMEM_LIB      "$NVSHMEM_LIB"
check_dir  VENV_PATH         "$VENV_PATH"
check_file CONTAINER_PATH   "$CONTAINER_PATH"
check_file BENCHMARK         "$BENCHMARK"
check_file PROFILER_PATH    "$PROFILER_PATH"

$ok && echo "All paths OK."
