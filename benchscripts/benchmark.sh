#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/library.sh"

usage() {
    echo "Usage: $0 --name=<benchmark> [args...]"
    echo ""
    echo "Benchmarks: cholesky, lu"
    exit 1
}

NAME=""
REMAINING=()
for arg in "$@"; do
    case "$arg" in
        --name=*) NAME="${arg#--name=}" ;;
        *) REMAINING+=("$arg") ;;
    esac
done

[[ -z "$NAME" ]] && { echo "ERROR: --name required"; usage; }

case "$NAME" in
    cholesky) BINARY="${BIN_DIR}/cholesky_mustard" ;;
    lu)       BINARY="${BIN_DIR}/lu_mustard" ;;
    *) echo "ERROR: unknown benchmark '$NAME'"; usage ;;
esac

exec "$BINARY" "${REMAINING[@]}"
