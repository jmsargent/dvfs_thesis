#!/bin/bash
# Usage: ./extract_freqs.sh <file> <n>
# Extracts every nth core clock frequency from the nvidia-smi clock query output.
# ../jobs/077-query-l4-clocks/slurm-172829.out
FILE="${1}"
N="${2:-1}"

FREQS=($(awk '
  /=== Supported Graphics \(SM\) Clock Frequencies/ { in_section=1; next }
  /=== / { in_section=0 }
  in_section && /^[0-9]+ MHz$/ { print $1 }
' "$FILE" | awk -v n="$N" 'NR % n == 1'))

echo "FREQS=(${FREQS[*]})"
