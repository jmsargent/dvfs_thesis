#!/usr/bin/env python3
"""Extract cholesky benchmark timings from L28 parameter search output."""

# python extract_l28_cholesky.py /data/users/sargent/dvfs_thesis/local_jobs/L28-find-parameters-to-use/result_186289.out

import re
import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

def parse(path):
    header_re = re.compile(r'--- cholesky n=(\d+) t=(\d+) streams=(\d+) ---')
    run_re    = re.compile(r'device (\d+) \| (\d+) run \| time \(s\): ([\d.]+)')

    rows = []
    current = None

    with open(path) as f:
        for line in f:
            m = header_re.search(line)
            if m:
                current = dict(n=int(m[1]), t=int(m[2]), streams=int(m[3]))
                continue

            if current is None:
                continue

            m = run_re.search(line)
            if m:
                rows.append({**current, 'device': int(m[1]), 'run': int(m[2]), 'time_s': float(m[3])})

    return pd.DataFrame(rows, columns=['n', 't', 'streams', 'device', 'run', 'time_s'])

def load(path):
    return parse(path)


# returns  (N, T, S, run, max(device execution time))
def slowest(df):
    return (
        df.groupby(['n', 't', 'streams', 'run'], sort=False)['time_s']
        .max()
        .reset_index()
        .rename(columns={'n': 'N', 't': 'T', 'streams': 'S', 'time_s': 'slowest_device_time_s'})
    )

# This is the variance of the slowest device for each run for each N,T,S
def check_variance_slowest(df):
    """Print per-(N,T,S) spread of slowest-device time across runs as CV%."""
    s = slowest(df)
    stats = (
        s.groupby(['N', 'T', 'S'])['slowest_device_time_s']
        .agg(mean='mean', std='std')
        .assign(cv_pct=lambda x: 100 * x['std'] / x['mean'])
        .reset_index()
    )
    print(stats.to_string(index=False, float_format='%.2f'))
    return stats

def plot_total_time_bar(df, out_path='cholesky_total_time.pdf'):
    tt = total_time(df)
    labels = [f"({int(row['N'])},{int(row['T'])},{int(row['S'])})" for _, row in tt.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, tt['total_time_s'])
    ax.set_xlabel('(N, T, S)')
    ax.set_ylabel('Total time (s)')
    ax.set_title('Cholesky: total time for 5 runs')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def save_variance_table(df, out_path='cholesky_variance.pdf'):
    stats = check_variance_slowest(df)
    stats = stats.round({'mean': 2, 'std': 2, 'cv_pct': 2})
    col_labels = ['N', 'T', 'S', 'mean (s)', 'std (s)', 'CV (%)']
    cell_text = stats.values.tolist()

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(cell_text) + 1))
    ax.axis('off')
    ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_t_sweep(df, out_dir='.'):
    """Plot mean run time vs T for the T sweep (N=24000, S=1), save as PGF and CSV."""
    tt = total_time(df)
    sweep = tt[(tt['S'] == 1) & (tt['N'] == tt['N'].max())].copy()
    sweep['mean_s'] = sweep['total_time_s'] / 5

    out = pathlib.Path(out_dir)
    sweep.to_csv(out / 'cholesky_t_sweep.csv', index=False)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(sweep['T'], sweep['mean_s'], marker='o')
    ax.set_xlabel('Tile size $T$')
    ax.set_ylabel('Mean run time (s)')
    ax.set_title('Cholesky: tile size sweep ($N=24000$, $S=1$)')
    fig.tight_layout()
    fig.savefig(out / 'cholesky_t_sweep.pdf')
    plt.close(fig)
    print(f"Saved: {out / 'cholesky_t_sweep.pdf'} and cholesky_t_sweep.csv")

def total_time(df):
    return (
        slowest(df)
        .groupby(['N', 'T', 'S'])['slowest_device_time_s']
        .sum()
        .reset_index()
        .rename(columns={'slowest_device_time_s': 'total_time_s'})
    )

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_file>")
        sys.exit(1)

    df = load(sys.argv[1])

    print("\n=== Total time per (N, T, S) ===")
    print(total_time(df).to_string(index=False, float_format='%.2f'))

    print("\n=== Variance of slowest device time across runs ===")
    check_variance_slowest(df)

    # plot_t_sweep(df, out_dir='/data/users/sargent/dvfs_thesis/result-processing/cholesky_sweep')
