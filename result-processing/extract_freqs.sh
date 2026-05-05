#!/bin/bash
# Usage: ./extract_freqs.sh <file> <mod> [offset]
# Extracts frequencies at indices where (index + offset) % mod == 0.
# ../jobs/077-query-l4-clocks/slurm-172829.out


# Examples:
# ./extract_freqs.sh file.out 3 0 → indices 0, 3, 6, ... (41 freqs)
# ./extract_freqs.sh file.out 3 1 → indices 1, 4, 7, ... (41 freqs)
# ./extract_freqs.sh file.out 3 2 → indices 2, 5, 8, ... (41 freqs)
#

FILE="${1}"
MOD="${2:-1}"
OFFSET="${3:-0}"

mapfile -t FREQS < <(awk '
  /=== Supported Graphics \(SM\) Clock Frequencies/ { in_section=1; next }
  /=== / { in_section=0 }
  in_section && /^[0-9]+ MHz$/ { print $1 }
' "$FILE" | awk -v mod="$MOD" -v offset="$OFFSET" '(NR - 1 + offset) % mod == 0')

echo "FREQS=(${FREQS[*]})"


# ./extract_freqs.sh ../jobs/077-query-l4-clocks/slurm-172829.out 3 0