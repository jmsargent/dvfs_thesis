#!/usr/bin/env python3
"""Extract benchmark timings from L28 parameter search output."""

# python extract_l28_cholesky.py <result_file> [benchmark]
# benchmark defaults to 'cholesky'; pass 'lu' for LU results
# local_jobs/L28-find-parameters-to-use/result.out
import re
import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
})

_BAR_FONTS = {
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
}

_SWEEP_FONTS = {
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
}

def parse(path, benchmark='cholesky'):
    header_re = re.compile(rf'--- {benchmark} n=(\d+) t=(\d+) streams=(\d+) ---')
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

def load(path, benchmark='cholesky'):
    return parse(path, benchmark)


# returns  (N, T, S, run, max(device execution time))
def slowest(df):
    return (
        df.groupby(['n', 't', 'streams', 'run'], sort=False)['time_s']
        .max()
        .reset_index()
        .rename(columns={'n': 'N', 't': 'T', 'streams': 'S', 'time_s': 'slowest_device_time_s'})
    )

def total_time(df):
    return (
        slowest(df)
        .groupby(['N', 'T', 'S'])['slowest_device_time_s']
        .sum()
        .reset_index()
        .rename(columns={'slowest_device_time_s': 'total_time_s'})
    )

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

def plot_total_time_bar(df, benchmark='cholesky', out_path=None):
    if out_path is None:
        out_path = f'{benchmark}_total_time.pgf'
    tt = total_time(df)
    labels = [f"({int(row['N'])},{int(row['T'])},{int(row['S'])})" for _, row in tt.iterrows()]

    with plt.rc_context(_BAR_FONTS):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(labels, tt['total_time_s'])
        ax.axhline(tt['total_time_s'].min(), color='red', linestyle='--', linewidth=1, label=f"min = {tt['total_time_s'].min():.1f} s")
        ax.legend()
        ax.set_xlabel('(N, tiles, streams)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'{benchmark.capitalize()} execution-times')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(out_path, backend='pgf', bbox_inches='tight')
        fig.savefig(pathlib.Path(out_path).with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
    print(f"Saved: {out_path}")

def save_variance_table(df, benchmark='cholesky', out_path=None):
    if out_path is None:
        out_path = f'{benchmark}_variance.tex'
    stats = check_variance_slowest(df)
    stats = stats.round({'mean': 2, 'std': 2, 'cv_pct': 2})

    lines = [
        r'\begin{tabular}{rrrrrr}',
        r'\toprule',
        r'$N$ & $T$ & $S$ & mean (s) & std (s) & CV (\%) \\',
        r'\midrule',
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"{int(row['N'])} & {int(row['T'])} & {int(row['S'])} "
            f"& {row['mean']:.2f} & {row['std']:.2f} & {row['cv_pct']:.2f} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}']

    pathlib.Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f"Saved: {out_path}")

def t_sweep(df):
    """Return total_time rows for the N with the most distinct T values at S=1."""
    tt = total_time(df)
    s1 = tt[tt['S'] == 1]
    n_sweep = s1.groupby('N')['T'].nunique().idxmax()
    return s1[s1['N'] == n_sweep].copy()

def plot_t_sweep(df, benchmark='cholesky', out_dir='.'):
    """Plot mean run time vs T for the T sweep (N with most T values at S=1), save as PGF and CSV."""
    sweep = t_sweep(df)
    sweep['mean_s'] = sweep['total_time_s'] / 5

    out = pathlib.Path(out_dir)
    sweep.to_csv(out / f'{benchmark}_t_sweep.csv', index=False)

    n_max = int(sweep['N'].iloc[0])
    with plt.rc_context(_SWEEP_FONTS):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep['T'], sweep['mean_s'], marker='o')
        ax.set_xlabel('Tiles $T$')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Tiles vs. executiontime: {benchmark.capitalize()}, N={n_max}')
        fig.tight_layout()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.savefig(out / f'{benchmark}_t_sweep.pgf', backend='pgf')
        fig.savefig(out / f'{benchmark}_t_sweep.pdf')
        plt.close(fig)
    print(f"Saved: {out / f'{benchmark}_t_sweep.pgf'} and {benchmark}_t_sweep.csv")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_file> [benchmark]")
        sys.exit(1)

    benchmark = sys.argv[2] if len(sys.argv) > 2 else 'cholesky'
    df = load(sys.argv[1], benchmark)

    print(f"\n=== {benchmark}: Total time per (N, T, S) ===")
    print(total_time(df).to_string(index=False, float_format='%.2f'))

    print(f"\n=== {benchmark}: Variance of slowest device time across runs ===")
    check_variance_slowest(df)

    out_dir = pathlib.Path('sweep') / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_total_time_bar(df, benchmark, out_path=out_dir / f'{benchmark}_total_time.pgf')
    save_variance_table(df, benchmark, out_path=out_dir / f'{benchmark}_variance.tex')
    plot_t_sweep(df, benchmark, out_dir=out_dir)