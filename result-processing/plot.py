import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path

def plot_energy_over_time(input_dir, save_dir):
    """
    Plots energy (Joules) per GPU over time for each unique experiment.
    Input files expected: bench_fFreq_tTile_gpuID.csv
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get all extracted CSV files
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Group files by experiment (Benchmark + Freq + Tile)
    # This ensures we plot all 4 GPUs for a single run on one graph
    experiments = {}
    for f in all_files:
        # Pattern matches the naming convention used in the previous extraction script
        match = re.search(r"(.+)_f(\d+)_t(\d+)_gpu(\d+).csv", os.path.basename(f))
        if match:
            # key: bench_fFreq_tTile
            exp_id = f"{match.group(1)}_f{match.group(2)}_t{match.group(3)}"
            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(f)

    for exp_id, files in experiments.items():
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        cmap = plt.get_cmap('tab10')

        # Extract metadata from exp_id for the title
        # lu_mustard_f795_t16 -> Bench: lu_mustard, Freq: 795, T: 16
        parts = exp_id.split('_f')
        bench_name = parts[0]
        freq_tile = parts[1].split('_t')
        freq = freq_tile[0]
        tile = freq_tile[1]

        for i, filepath in enumerate(sorted(files)):
            # Determine GPU ID from filename
            gpu_match = re.search(r"gpu(\d+)", os.path.basename(filepath))
            gpu_id = gpu_match.group(1) if gpu_match else i

            df = pd.read_csv(filepath)
            
            # 1. Normalize Time (Start at 0 seconds)
            df['elapsed_s'] = df['timestamp'] - df['timestamp'].min()
            
            # 2. Normalize Energy (mJ to J, start at 0 Joules)
            # Use the cumulative hardware counter
            df['energy_j'] = (df['total_energy_mj'] - df['total_energy_mj'].iloc[0]) / 1000.0

            # 3. Plotting
            ax.plot(df['elapsed_s'], df['energy_j'], 
                    label=f'GPU {gpu_id}', 
                    color=cmap(i % 10), 
                    linewidth=2, 
                    alpha=0.8)

        # Formatting
        plt.xlabel('Elapsed Time [s]', fontsize=12)
        plt.ylabel('Accumulated Energy [J]', fontsize=12)
        plt.title(f'Energy Consumption | {bench_name} | Freq={freq}MHz | T={tile}', fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(title="Devices", loc='best')
        plt.tight_layout()

        # plt.show()

        # Save and Close
        save_name = f"{exp_id}_energy_plot.png"
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)
        plt.close()
        print(f"Generated plot: {save_name}")

if __name__ == "__main__":
    # Point this to the directory where your parsed CSVs are stored
    EXTRACTED_DATA = "extracted_results"
    PLOT_OUTPUT = "energy_comparison_plots"
    
    plot_energy_over_time(EXTRACTED_DATA, PLOT_OUTPUT)