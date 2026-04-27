#!/bin/bash
# Replaces commas inside parentheses with semicolons in all CSV files in a directory.
# Usage: fix_csv_parens.sh [directory]
set -euo pipefail

dir="${1:-.}"

shopt -s nullglob
files=("$dir"/*.csv)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No CSV files found in: $dir" >&2
    exit 1
fi

perl -i -pe 's/\(([^)]+)\)/my $x=$1; $x=~s!,!;!g; "($x)"/ge' "${files[@]}"
echo "Processed ${#files[@]} file(s) in $dir"
