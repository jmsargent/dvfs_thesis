#!/usr/bin/env python3
"""Parse and plot benchmark results from run.sbatch output."""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def gflops_lu(n, t):
    return (2/3 * n**3 * 1e-9) / t

def gflops_chol(n, t):
    return (1/3 * n**3 * 1e-9) / t


def parse_file(path):
    records = []
    current_bench = None
    current_sweep = None
    mustard_starts = []
    mustard_ends = []
    slate_times = []
    starpu_gflops = []

    def flush():
        if current_bench is None:
            return
        solver = current_bench['solver']
        N = current_bench['N']
        is_lu = 'lu' in solver

        if 'mustard' in solver:
            times = [e - s for s, e in zip(mustard_starts, mustard_ends)]
            gf = [gflops_lu(N, t) if is_lu else gflops_chol(N, t) for t in times]
        elif 'slate' in solver:
            times = slate_times[:]
            gf = [gflops_lu(N, t) if is_lu else gflops_chol(N, t) for t in times]
        else:
            times = []
            gf = starpu_gflops[:]

        records.append({
            'solver': solver,
            'N': N,
            'NB': current_bench['NB'],
            'sweep': current_sweep,
            'times_sec': times,
            'gflops': gf,
        })

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            if 'SWEEP 1' in line:
                current_sweep = 'N_sweep'
                continue
            if 'SWEEP 2' in line:
                current_sweep = 'tile_sweep'
                continue

            m = re.match(r'BENCH idx=\d+ solver=(\S+) N=(\d+) NB=(\d+) runs=(\d+)', line)
            if m:
                flush()
                current_bench = {
                    'solver': m.group(1),
                    'N': int(m.group(2)),
                    'NB': int(m.group(3)),
                    'runs': int(m.group(4)),
                }
                mustard_starts = []
                mustard_ends = []
                slate_times = []
                starpu_gflops = []
                continue

            if current_bench is None:
                continue

            solver = current_bench['solver']

            if 'mustard' in solver:
                m = re.match(r'(?:lu|cholesky) tiledPanel start_time: (\S+)', line)
                if m:
                    mustard_starts.append(float(m.group(1)))
                    continue
                m = re.match(r'(?:lu|cholesky) tiledPanel end_time: (\S+)', line)
                if m:
                    mustard_ends.append(float(m.group(1)))
                    continue

            elif 'slate' in solver:
                m = re.search(r'time (\d+\.\d+) sec', line)
                if m:
                    slate_times.append(float(m.group(1)))
                    continue

            elif 'starpu' in solver:
                m = re.match(r'(\d+)\s+(\d+)\s+(\S+)$', line)
                if m and int(m.group(1)) == current_bench['N']:
                    starpu_gflops.append(float(m.group(3)))
                    continue

    flush()
    return records


STYLE = {
    'mustard_lu':   {'color': 'tab:blue',   'marker': 'o', 'label': 'mustard LU'},
    'mustard_chol': {'color': 'tab:blue',   'marker': 's', 'label': 'mustard Chol', 'linestyle': '--'},
    'slate_lu':     {'color': 'tab:orange', 'marker': 'o', 'label': 'SLATE LU'},
    'slate_chol':   {'color': 'tab:orange', 'marker': 's', 'label': 'SLATE Chol', 'linestyle': '--'},
    'starpu_lu':    {'color': 'tab:green',  'marker': 'o', 'label': 'StarPU LU'},
    'starpu_chol':  {'color': 'tab:green',  'marker': 's', 'label': 'StarPU Chol', 'linestyle': '--'},
}


def plot_sweep(records, x_key, x_label, title, ax):
    by_solver = defaultdict(list)
    for r in records:
        if r['gflops']:
            by_solver[r['solver']].append(r)

    for solver, recs in sorted(by_solver.items()):
        recs.sort(key=lambda r: r[x_key])
        xs = [r[x_key] for r in recs]
        medians = [np.median(r['gflops']) for r in recs]
        lo = [np.median(r['gflops']) - min(r['gflops']) for r in recs]
        hi = [max(r['gflops']) - np.median(r['gflops']) for r in recs]
        style = STYLE.get(solver, {'color': 'black', 'marker': 'x', 'label': solver})
        ax.errorbar(xs, medians, yerr=[lo, hi],
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style.get('linestyle', '-'),
                    label=style['label'],
                    capsize=4, linewidth=1.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel('GFlop/s')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} result_*.out [output.pdf]')
        sys.exit(1)

    path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else Path(path).stem + '_plots.pdf'

    records = parse_file(path)
    if not records:
        print('No records parsed — check the file format.')
        sys.exit(1)

    n_sweep    = [r for r in records if r['sweep'] == 'N_sweep']
    tile_sweep = [r for r in records if r['sweep'] == 'tile_sweep']

    nrows = sum([bool(n_sweep), bool(tile_sweep)])
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 5 * nrows), squeeze=False)
    fig.suptitle(Path(path).name)

    row = 0
    if n_sweep:
        nb = n_sweep[0]['NB']
        plot_sweep(n_sweep, 'N', 'Matrix size N',
                   f'N sweep (NB={nb})', axes[row][0])
        row += 1

    if tile_sweep:
        n = tile_sweep[0]['N']
        plot_sweep(tile_sweep, 'NB', 'Tile size NB',
                   f'Tile size sweep (N={n})', axes[row][0])

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
