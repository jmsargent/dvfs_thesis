#!/usr/bin/env python3
"""
Parse benchmark output file and produce a per-task energy/timing CSV.

Each row in the output is one (bench, freq, pe, task) combination,
with timing and energy values averaged across runs.

Energy is attributed by interpolating the per-GPU cumulative energy
time-series at each task's wait_start_ts, start_ts, and end_ts.

Output columns:
    bench, freq_mhz, pe, task_id, op_name,
    calc_time_ns, wait_time_ns, calc_energy_mj, wait_energy_mj

Usage:
    python parse_benchmark.py <input_file> <output_file>
"""

import sys
import csv
import re
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_file(input_path):
    """
    Stream-parse the benchmark output file.

    Returns
    -------
    task_data : dict[(bench, freq)] -> list of task-row dicts
    energy_data : dict[(bench, freq, gpu)] -> list of (timestamp_s, total_energy_mj)
    """
    task_data = defaultdict(list)
    energy_data = defaultdict(list)

    current_bench = None
    current_freq = None
    in_task_section = False
    in_energy_section = False
    energy_header_seen = False
    current_gpu = None

    with open(input_path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # ---- experiment header ------------------------------------------
            m = re.match(r"START_EXPERIMENT: BENCH=(\S+?)_FREQ=(\d+)_", line)
            if m:
                current_bench = m.group(1)
                current_freq = int(m.group(2))
                in_task_section = False
                in_energy_section = False
                continue

            # ---- task CSV header --------------------------------------------
            if line == "pe,run,task_id,op_name,start_ts,end_ts,wait_start_ts":
                in_task_section = True
                in_energy_section = False
                continue

            # ---- energy section begin ---------------------------------------
            m = re.match(r"--- BEGIN CSV_DATA\|[^|]+\|(\d+)\|\d+\|(\d+) ---", line)
            if m:
                in_task_section = False
                in_energy_section = True
                energy_header_seen = False
                current_gpu = int(m.group(2))
                continue

            # ---- energy section end -----------------------------------------
            if line.startswith("--- END CSV_DATA|"):
                in_energy_section = False
                current_gpu = None
                continue

            # ---- task data row ----------------------------------------------
            if in_task_section and current_bench is not None:
                # op_name (e.g. "GEMM(1,3,0)") may contain commas, so we
                # anchor on the first 3 and last 3 comma-separated fields.
                parts = line.split(",")
                if len(parts) >= 7:
                    try:
                        row = {
                            "pe":            int(parts[0]),
                            "run":           int(parts[1]),
                            "task_id":       int(parts[2]),
                            "op_name":       ",".join(parts[3:-3]),
                            "start_ts":      int(parts[-3]),
                            "end_ts":        int(parts[-2]),
                            "wait_start_ts": int(parts[-1]),
                        }
                        task_data[(current_bench, current_freq)].append(row)
                    except ValueError:
                        pass  # corrupted line (interleaved MPI stdout), skip
                continue

            # ---- energy data row --------------------------------------------
            if in_energy_section and current_gpu is not None:
                if not energy_header_seen:
                    if line.startswith("timestamp,"):
                        energy_header_seen = True
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        ts = float(parts[0])
                        total_energy = float(parts[3])
                        energy_data[(current_bench, current_freq, current_gpu)].append(
                            (ts, total_energy)
                        )
                    except ValueError:
                        pass

    return task_data, energy_data


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def interp_energy(energy_array, ts_ns):
    """
    Interpolate cumulative energy (mJ) at nanosecond timestamp ts_ns.

    energy_array : np.ndarray of shape (N, 2) — columns: [timestamp_s, total_energy_mj]
    """
    ts_s = ts_ns / 1e9
    return float(np.interp(ts_s, energy_array[:, 0], energy_array[:, 1]))


def process(task_data, energy_data):
    """
    Compute per-task averaged metrics across runs.

    Returns a list of output row dicts.
    """
    rows = []

    for (bench, freq), tasks in task_data.items():
        # Pre-build energy arrays per GPU for this experiment.
        energy_arrays = {}
        for gpu in range(4):
            series = energy_data.get((bench, freq, gpu))
            if series:
                arr = np.array(series, dtype=np.float64)
                # Sort by timestamp in case samples arrive out of order.
                arr = arr[arr[:, 0].argsort()]
                energy_arrays[gpu] = arr

        # Group task rows by (pe, task_id); keep op_name from first occurrence.
        groups = defaultdict(list)
        op_names = {}
        for t in tasks:
            key = (t["pe"], t["task_id"])
            groups[key].append(t)
            op_names.setdefault(key, t["op_name"])

        for (pe, task_id), run_rows in groups.items():
            ea = energy_arrays.get(pe)

            calc_times = []
            wait_times = []
            calc_energies = []
            wait_energies = []

            for t in run_rows:
                start_ts     = t["start_ts"]
                end_ts       = t["end_ts"]
                wait_start_ts = t["wait_start_ts"]

                calc_time = end_ts - start_ts
                calc_times.append(calc_time)

                # wait_start_ts == 0 means the task had no dependency to wait for.
                if wait_start_ts == 0:
                    wait_times.append(0)
                else:
                    wait_times.append(start_ts - wait_start_ts)

                if ea is not None:
                    ce = interp_energy(ea, end_ts) - interp_energy(ea, start_ts)
                    calc_energies.append(ce)

                    if wait_start_ts == 0:
                        wait_energies.append(0.0)
                    else:
                        we = interp_energy(ea, start_ts) - interp_energy(ea, wait_start_ts)
                        wait_energies.append(we)

            n = len(run_rows)
            rows.append({
                "bench":          bench,
                "freq_mhz":       freq,
                "pe":             pe,
                "task_id":        task_id,
                "op_name":        op_names[(pe, task_id)],
                "calc_time_ns":   sum(calc_times) / n,
                "wait_time_ns":   sum(wait_times) / n,
                "calc_energy_mj": sum(calc_energies) / len(calc_energies) if calc_energies else "",
                "wait_energy_mj": sum(wait_energies) / len(wait_energies) if wait_energies else "",
            })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "bench", "freq_mhz", "pe", "task_id", "op_name",
    "calc_time_ns", "wait_time_ns", "calc_energy_mj", "wait_energy_mj",
]


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Parsing {input_path} ...", file=sys.stderr)
    task_data, energy_data = parse_file(input_path)

    total_task_rows = sum(len(v) for v in task_data.values())
    print(f"  {total_task_rows} task rows across "
          f"{len(task_data)} (bench, freq) combinations", file=sys.stderr)

    print("Processing ...", file=sys.stderr)
    rows = process(task_data, energy_data)
    print(f"  {len(rows)} output rows", file=sys.stderr)

    print(f"Writing {output_path} ...", file=sys.stderr)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.", file=sys.stderr)

def print_wait_times():
    pass

if __name__ == "__main__":
    main()
