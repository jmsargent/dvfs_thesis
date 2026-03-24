import os
import re
import csv
from io import StringIO

def parse_slurm_output(file_path, output_dir="extracted_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Regex to catch: CSV_DATA|BENCH|FREQ|TILE|GPU_ID
    pattern = re.compile(r"--- BEGIN CSV_DATA\|(?P<bench>.*?)\|(?P<freq>\d+)\|(?P<tile>\d+)\|(?P<gpu>\d+) ---")

    with open(file_path, 'r') as f:
        content = f.read()

    # Find all matches
    matches = list(pattern.finditer(content))

    for i, match in enumerate(matches):
        meta = match.groupdict()
        start_index = match.end()

        # Find the corresponding END tag
        end_tag = f"--- END CSV_DATA|{meta['bench']}|{meta['freq']}|{meta['tile']}|{meta['gpu']} ---"
        end_index = content.find(end_tag, start_index)

        if end_index != -1:
            csv_data = content[start_index:end_index].strip()

            # Construct a clean filename
            filename = f"{meta['bench']}_f{meta['freq']}_t{meta['tile']}_gpu{meta['gpu']}.csv"
            save_path = os.path.join(output_dir, filename)

            with open(save_path, 'w') as out_f:
                out_f.write(csv_data)

            print(f"Extracted: {filename}")

if __name__ == "__main__":
    # Replace with your actual Slurm output filename
    parse_slurm_output("/data/users/sargent/dvfs_thesis/jobs/041-experiment-4gpus-baseline-2040mhz-n4800-t16-r5/slurm-142371.out")