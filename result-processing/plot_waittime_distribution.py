import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATABASES = {
    'cholesky': 'cholesky_database.csv',
    'lu':       'lu_database.csv',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (name, path) in enumerate(DATABASES.items()):
    df = pd.read_csv(path)

    per_op = df.drop_duplicates(subset=['op_name', 'pe', 'freq_mhz'])['average_waittime_ns']
    per_op = per_op[per_op > 0] / 1_000  # µs

    # Log-scale histogram
    ax_hist = axes[0, col]
    log_bins = np.logspace(np.log10(per_op.min()), np.log10(per_op.max()), 60)
    ax_hist.hist(per_op, bins=log_bins, edgecolor='none', alpha=0.8)
    ax_hist.set_xscale('log')
    ax_hist.set_title(f'{name} — log-scale histogram')
    ax_hist.set_xlabel('Average wait time (µs, log scale)')
    ax_hist.set_ylabel('Count')

    # CDF
    ax_cdf = axes[1, col]
    sorted_vals = np.sort(per_op)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax_cdf.plot(sorted_vals, cdf)
    ax_cdf.set_xscale('log')
    ax_cdf.set_title(f'{name} — CDF')
    ax_cdf.set_xlabel('Average wait time (µs, log scale)')
    ax_cdf.set_ylabel('Cumulative fraction')
    ax_cdf.grid(True, alpha=0.3)

    print(f"\n=== {name} wait time stats (µs) ===")
    print(per_op.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99]).to_string())
    print(f"variance:  {per_op.var():.1f} µs²")

fig.tight_layout()
plt.show()

# --- Per-kernel breakdown ---
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

for col, (name, path) in enumerate(DATABASES.items()):
    df = pd.read_csv(path)
    df['kernel'] = df['op_name'].str.extract(r'^([A-Za-z_]+)', expand=False)

    per_op = df.drop_duplicates(subset=['op_name', 'pe', 'freq_mhz'])
    per_op = per_op[per_op['average_waittime_ns'] > 0].copy()
    per_op['wait_us'] = per_op['average_waittime_ns'] / 1_000

    ax_hist = axes2[0, col]
    ax_cdf  = axes2[1, col]

    for kernel, group in per_op.groupby('kernel'):
        vals = group['wait_us'].to_numpy()
        log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 50)
        ax_hist.hist(vals, bins=log_bins, alpha=0.5, label=kernel)

        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, label=kernel)

    ax_hist.set_xscale('log')
    ax_hist.set_title(f'{name} — wait time per kernel')
    ax_hist.set_xlabel('Average wait time (µs, log scale)')
    ax_hist.set_ylabel('Count')
    ax_hist.legend()

    ax_cdf.set_xscale('log')
    ax_cdf.set_title(f'{name} — CDF per kernel')
    ax_cdf.set_xlabel('Average wait time (µs, log scale)')
    ax_cdf.set_ylabel('Cumulative fraction')
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend()

fig2.tight_layout()
plt.show()
