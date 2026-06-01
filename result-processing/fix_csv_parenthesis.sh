#!/bin/bash
# Replaces commas inside parentheses with semicolons in CSV files found under
# subdirectories whose name matches a given pattern.
# Usage: fix_csv_parens.sh [directory] [subdir-name]
set -euo pipefail

dir="${1:-.}"
subdir_name="${2:?Usage: $0 [directory] <subdir-name>}"

files=()
while IFS= read -r -d '' f; do
    files+=("$f")
done < <(find "$dir" -type d -name "$subdir_name" -print0 \
    | xargs -0 -I{} find {} -maxdepth 1 -type f -name "*.csv" -print0)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No CSV files found under '$subdir_name' directories in: $dir" >&2
    exit 1
fi

perl -i -pe 's/\(([^)]+)\)/my $x=$1; $x=~s!,!;!g; "($x)"/ge' "${files[@]}"
echo "Processed ${#files[@]} file(s) from '$subdir_name' directories under $dir"
