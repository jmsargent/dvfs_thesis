import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import json

def trim_to_active_window(df):
    """Trim dataframe to rows where GPU is actively computing.

    Uses gpu_util_pct > 0 to detect the compute window. Power is not a
    reliable signal on L4s because idle (~22W) and active (~26W) are too
    close together."""
    active = df[df['gpu_util_pct'] > 0]
    if active.empty:
        return df
    return df.loc[active.index[0]:active.index[-1]]


def plot_energy_over_time(input_dir, save_dir):
    """
    Plots energy (Joules) per GPU over time for each unique experiment.
    Input files expected: bench_fFreq_tTile_gpuID.csv
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        cmap = plt.get_cmap('tab10')

        parts = exp_id.split('_f')
        bench_name = parts[0]
        freq_tile = parts[1].split('_t')
        freq = freq_tile[0]
        tile = freq_tile[1]

        for i, filepath in enumerate(sorted(files)):
            gpu_match = re.search(r"gpu(\d+)", os.path.basename(filepath))
            gpu_id = gpu_match.group(1) if gpu_match else i

            df = pd.read_csv(filepath)

            # Trim to active compute window using power threshold
            df = trim_to_active_window(df)

            # Normalize time (start at 0)
            df['elapsed_s'] = df['timestamp'] - df['timestamp'].min()

            # Normalize energy (mJ to J, start at 0)
            df['energy_j'] = (df['total_energy_mj'] - df['total_energy_mj'].iloc[0]) / 1000.0

            ax.plot(df['elapsed_s'], df['energy_j'],
                    label=f'GPU {gpu_id}',
                    color=cmap(i % 10),
                    linewidth=2,
                    alpha=0.8)

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
        plt.legend(title="Devices", loc='best')
        plt.tight_layout()

        save_name = f"{exp_id}_energy_plot.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close()
        print(f"Generated plot: {save_name}")


if __name__ == "__main__":
    EXTRACTED_DATA = "extracted_results"
    PLOT_OUTPUT = "energy_comparison_plots2"

    plot_energy_over_time(EXTRACTED_DATA, PLOT_OUTPUT)