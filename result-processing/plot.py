import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import json
from datetime import datetime


def trim_to_active_window(df):
    """Trim dataframe to rows where GPU is actively computing.

    Uses gpu_util_pct > 0 to detect the compute window. Power is not a
    reliable signal on L4s because idle (~22W) and active (~26W) are too
    close together."""
    active = df[df['gpu_util_pct'] > 0]
    if active.empty:
        return df
    return df.loc[active.index[0]:active.index[-1]]


def parse_slurm_timings(slurm_path):
    """Parse phase timings per experiment from slurm output file.

    Returns dict keyed by exp_id (e.g. 'cholesky_mustard_f795_t16').
    Uses device 0 as the reference for all timers.
    Raises ValueError if any timing field is missing for an experiment.
    """
    timings = {}
    current_exp = None

    with open(slurm_path) as f:
        for line in f:
            line = line.strip()

            m = re.match(r'START_EXPERIMENT: BENCH=(.+?)_FREQ=(\d+)_TILE=(\d+)', line)
            if m:
                bench, freq, tile = m.group(1), m.group(2), m.group(3)
                current_exp = f"{bench}_f{freq}_t{tile}"
                timings[current_exp] = {
                    'nvshmem_init': None,
                    'setup': None,
                    'total_calc': None,
                    'total_program': None,
                    'program_start_ts': None,
                    'program_end_ts': None,
                }
                continue

            if current_exp is None:
                continue

            t = timings[current_exp]

            m = re.match(r'device 0 \| NVSHMEM init time \(s\): ([\d.]+)', line)
            if m:
                t['nvshmem_init'] = float(m.group(1))
                continue

            m = re.match(r'device 0 \| Setup time \(s\): ([\d.]+)', line)
            if m:
                t['setup'] = float(m.group(1))
                continue

            m = re.match(r'Total time used \(s\): ([\d.]+)', line)
            if m:
                t['total_calc'] = float(m.group(1))
                continue

            m = re.match(r'device 0 \| Total program time \(s\): ([\d.]+)', line)
            if m:
                t['total_program'] = float(m.group(1))
                continue

            m = re.match(r'Program start timestamp: ([\d.]+)', line)
            if m:
                t['program_start_ts'] = float(m.group(1))
                continue

            m = re.match(r'Program end timestamp: ([\d.]+)', line)
            if m:
                t['program_end_ts'] = float(m.group(1))

    for exp_id, t in timings.items():
        missing = [k for k, v in t.items() if v is None]
        if missing:
            raise ValueError(f"Missing timing fields {missing} for experiment '{exp_id}'")

    return timings


def plot_energy_over_time(input_dir, save_dir, slurm_path):
    """
    Plots energy (Joules) per GPU over time for each unique experiment,
    with shaded regions for program phases parsed from the slurm output.
    Input files expected: bench_fFreq_tTile_gpuID.csv
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    slurm_timings = parse_slurm_timings(slurm_path)

    # Load CUDA event timings if available
    timing_path = os.path.join(input_dir, "cuda_event_timings.json")
    cuda_timings = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            cuda_timings = json.load(f)

    # Get all extracted CSV files
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Group files by experiment (Benchmark + Freq + Tile)
    experiments = {}
    for f in all_files:
        match = re.search(r"(.+)_f(\d+)_t(\d+)_gpu(\d+).csv", os.path.basename(f))
        if match:
            exp_id = f"{match.group(1)}_f{match.group(2)}_t{match.group(3)}"
            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(f)

    for exp_id, files in experiments.items():
        if exp_id not in slurm_timings:
            raise ValueError(f"No slurm timing data found for experiment: '{exp_id}'")

        phase = slurm_timings[exp_id]

        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        cmap = plt.get_cmap('tab10')

        parts = exp_id.split('_f')
        bench_name = parts[0]
        freq_tile = parts[1].split('_t')
        freq = freq_tile[0]
        tile = freq_tile[1]

        # Use gpu0's CSV to compute the anchor offset between program start and
        # the first active GPU moment (t=0 on the elapsed_s axis).
        gpu0_file = sorted(files)[0]
        df0_raw = pd.read_csv(gpu0_file)
        raw_csv_start = df0_raw['timestamp'].iloc[0]
        active_rows = df0_raw[df0_raw['gpu_util_pct'] > 0]
        first_active_ts = df0_raw.loc[active_rows.index[0], 'timestamp'] if not active_rows.empty else raw_csv_start
        # program_start_on_plot: where program start falls on the elapsed_s axis
        program_start_on_plot = phase['program_start_ts'] - first_active_ts

        for i, filepath in enumerate(sorted(files)):
            gpu_match = re.search(r"gpu(\d+)", os.path.basename(filepath))
            gpu_id = gpu_match.group(1) if gpu_match else i

            df = pd.read_csv(filepath)

            # Trim to active compute window
            df = trim_to_active_window(df)

            # Normalize time (start at 0 from first active moment)
            df['elapsed_s'] = df['timestamp'] - first_active_ts

            # Normalize energy (mJ to J, start at 0)
            df['energy_j'] = (df['total_energy_mj'] - df['total_energy_mj'].iloc[0]) / 1000.0

            ax.plot(df['elapsed_s'], df['energy_j'],
                    label=f'GPU {gpu_id}',
                    color=cmap(i % 10),
                    linewidth=2,
                    alpha=0.8)

        # Phase positions on the elapsed_s axis, anchored via the program start timestamp.
        nvshmem_end = program_start_on_plot + phase['nvshmem_init']
        setup_end   = nvshmem_end + phase['setup']
        calc_end    = setup_end   + phase['total_calc']

        phase_regions = [
            (program_start_on_plot, nvshmem_end, 'NVSHMEM Init', 'steelblue',   0.15),
            (nvshmem_end,   setup_end,   'CUDA Setup',   'darkorange',  0.15),
            (setup_end,     calc_end,    'Calculation',  'forestgreen', 0.15),
        ]

        for x_start, x_end, label, color, alpha in phase_regions:
            ax.axvspan(x_start, x_end, alpha=alpha, color=color, label=label)

        # Annotate with CUDA event timing if available
        subtitle = ""
        if exp_id in cuda_timings:
            t = cuda_timings[exp_id]
            runs = t["runs"]
            if runs:
                avg = sum(runs) / len(runs)
                subtitle = f"\nCUDA kernel time: avg={avg:.4f}s, total={t['total_s']}s ({len(runs)} runs)"

        plt.xlabel('Elapsed Time [s]', fontsize=12)
        plt.ylabel('Accumulated Energy [J]', fontsize=12)
        plt.title(f'Energy Consumption | {bench_name} | Freq={freq}MHz | T={tile}{subtitle}', fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(title="Devices / Phases", loc='best')
        plt.tight_layout()

        save_name = f"{exp_id}_energy_plot.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close()
        print(f"Generated plot: {save_name}")


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    EXTRACTED_DATA = "/Users/jonathansargent/dvfs_thesis/result-processing/extracted_results-27thmarch1352"
    PLOT_OUTPUT    = f"/Users/jonathansargent/dvfs_thesis/result-processing/energy_comparison_plots_{now}"
    SLURM_FILE     = "/Users/jonathansargent/dvfs_thesis/jobs/046-experiment-4gpus-baseline-2040mhz-n4800-t16-r5-wall-time-n9600/slurm-161652.out"

    plot_energy_over_time(EXTRACTED_DATA, PLOT_OUTPUT, SLURM_FILE)
