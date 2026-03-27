"""
Parse a Slurm output file from a mustard DVFS experiment.

Extracts two things:
  1. Per-GPU profiler CSVs (power, energy, frequency samples from EnergyMonitor.py)
  2. CUDA event kernel timings reported by the mustard benchmark itself

The Slurm output contains multiple experiments delimited by
START_EXPERIMENT / END_EXPERIMENT tags. Within each experiment:
  - Profiler CSV data is wrapped in BEGIN/END CSV_DATA tags (one per GPU)
  - Kernel timing lines look like: "device 0 | 2 run | time (s): 0.1234"
  - A summary line: "Total time used (s): 0.6170"
"""

import os
import re
import json


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches: --- BEGIN CSV_DATA|cholesky_mustard|795|16|0 ---
CSV_BEGIN_RE = re.compile(
    r"--- BEGIN CSV_DATA\|(?P<bench>.*?)\|(?P<freq>\d+)\|(?P<tile>\d+)\|(?P<gpu>\d+) ---"
)

# Matches: START_EXPERIMENT: BENCH=lu_mustard_FREQ=2040_TILE=16_N=4800_R=5
EXP_START_RE = re.compile(
    r"START_EXPERIMENT: BENCH=(?P<bench>\S+)_FREQ=(?P<freq>\d+)_TILE=(?P<tile>\d+)_N=(?P<n>\d+)_R=(?P<runs>\d+)"
)

# Matches: device 0 | 3 run | time (s): 0.1285
RUN_TIME_RE = re.compile(r"device \d+ \| (?P<run>\d+) run \| time \(s\): (?P<time>[\d.]+)")

# Matches: Total time used (s): 0.6643
TOTAL_TIME_RE = re.compile(r"Total time used \(s\): (?P<total>[\d.]+)")

EXP_END_RE = re.compile(r"END_EXPERIMENT:")


# ---------------------------------------------------------------------------
# CSV extraction — profiler power/energy/frequency samples
# ---------------------------------------------------------------------------

def extract_profiler_csvs(content, output_dir):
    """Find every BEGIN/END CSV_DATA block and write each one to a file.

    Files are named like: cholesky_mustard_f795_t16_gpu0.csv
    """
    for match in CSV_BEGIN_RE.finditer(content):
        meta = match.groupdict()

        # Locate the matching END tag
        end_tag = f"--- END CSV_DATA|{meta['bench']}|{meta['freq']}|{meta['tile']}|{meta['gpu']} ---"
        end_index = content.find(end_tag, match.end())
        if end_index == -1:
            print(f"Warning: no END tag found for {meta}, skipping")
            continue

        csv_data = content[match.end():end_index].strip()

        filename = f"{meta['bench']}_f{meta['freq']}_t{meta['tile']}_gpu{meta['gpu']}.csv"
        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'w') as f:
            f.write(csv_data)

        print(f"Extracted CSV: {filename}")


# ---------------------------------------------------------------------------
# Timing extraction — CUDA event times from the benchmark binary
# ---------------------------------------------------------------------------

def extract_cuda_timings(content):
    """Walk experiment blocks and collect per-run kernel times.

    Returns a dict keyed by experiment id (e.g. "lu_mustard_f795_t16"):
        {
            "n": 4800,
            "runs": [0.1439, 0.1317, ...],
            "total_s": 0.6643
        }
    """
    timings = {}
    current_exp = None

    for line in content.splitlines():
        # Enter an experiment block
        start_match = EXP_START_RE.search(line)
        if start_match:
            meta = start_match.groupdict()
            current_exp = f"{meta['bench']}_f{meta['freq']}_t{meta['tile']}"
            timings[current_exp] = {"n": int(meta["n"]), "runs": [], "total_s": None}
            continue

        if current_exp is None:
            continue

        # Per-run timing line
        run_match = RUN_TIME_RE.search(line)
        if run_match:
            timings[current_exp]["runs"].append(float(run_match.group("time")))
            continue

        # Total time summary line
        total_match = TOTAL_TIME_RE.search(line)
        if total_match:
            timings[current_exp]["total_s"] = float(total_match.group("total"))
            continue

        # Exit the experiment block
        if EXP_END_RE.search(line):
            current_exp = None

    return timings


def save_cuda_timings(timings, output_dir):
    """Write timings dict to JSON and print a summary."""
    timing_path = os.path.join(output_dir, "cuda_event_timings.json")
    with open(timing_path, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"Extracted timings: {timing_path}")

    for exp_id, data in timings.items():
        runs = data["runs"]
        if runs:
            avg = sum(runs) / len(runs)
            print(f"  {exp_id}: {len(runs)} runs, avg={avg:.4f}s, total={data['total_s']}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_slurm_output(file_path, output_dir="extracted_results-27thmarch1149"):
    """Top-level entry point: extract profiler CSVs and CUDA timings."""
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r') as f:
        content = f.read()

    extract_profiler_csvs(content, output_dir)

    timings = extract_cuda_timings(content)
    save_cuda_timings(timings, output_dir)


if __name__ == "__main__":
    parse_slurm_output(
        "/Users/jonathansargent/dvfs_thesis/jobs/043-experiment-4gpus-baseline-2040mhz-n4800-t16-r5-detailed-timers/slurm-160534.out"
    )
