import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

N_PES = 4

def classify_locality(op_name):
    """
    Returns 'local' or 'remote' based on 1D block-cyclic panel distribution
    with N_PES GPUs.

    Cholesky panel ownership:
      POTRF(k)                  → owned by k%P,          always local
      TRSM(column;pivotCol)     → owned by pivotCol%P,   always local
      SYRK(column;pivotCol)     → owned by column%P,     local iff column%P == pivotCol%P
      GEMM(row;column;pivotCol) → owned by column%P,     local iff column%P == pivotCol%P

    LU panel ownership:
       KERNEL         | 
      GETRF(k)        | k%P | always local
      TRSM_R(i;k)     | k%P | always local
      TRSM_L(k;i)     | i%P | i%P == k%P
      GEMM(i;j;k)     | j%P | j%P == k%P
    """
    op_type = op_name.split('(')[0]
    params_str = op_name[len(op_type)+1:].rstrip(')')
    params = [int(p) for p in params_str.split(';')] if params_str else []

    always_local = {'POTRF', 'GETRF', 'TRSM', 'TRSM_R'}
    if op_type in always_local:
        return 'local'

    if op_type == 'SYRK':
        return 'local' if params[0] % N_PES == params[1] % N_PES else 'remote'

    if op_type == 'GEMM':
        return 'local' if params[1] % N_PES == params[2] % N_PES else 'remote'

    if op_type == 'TRSM_L':
        return 'local' if params[1] % N_PES == params[0] % N_PES else 'remote'

    return 'unknown'


lu = pd.read_csv("lu_database.csv")
ch = pd.read_csv("cholesky_database.csv")

for name, df in [("LU", lu), ("Cholesky", ch)]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}  |  freq_mhz unique: {sorted(df['freq_mhz'].unique())}")
    nonzero_pct = 100 * df['average_waittime_ns'].gt(0).mean()
    print(f"average_waittime_ns: mean={df['average_waittime_ns'].mean():.1f}, "
          f"nonzero={df['average_waittime_ns'].gt(0).sum():,} ({nonzero_pct:.1f}%)")

    # --- Overall correlation (all rows) ---
    r_p, p_p = stats.pearsonr(df['freq_mhz'], df['average_waittime_ns'])
    r_s, p_s = stats.spearmanr(df['freq_mhz'], df['average_waittime_ns'])
    print("\nOverall correlation (all rows):")
    print(f"  Pearson  r = {r_p:+.4f}  p = {p_p:.2e}")
    print(f"  Spearman r = {r_s:+.4f}  p = {p_s:.2e}")

    # --- Correlation restricted to rows with nonzero waittime ---
    nz = df[df['average_waittime_ns'] > 0]
    if len(nz) > 1 and nz['freq_mhz'].nunique() > 1:
        r_p2, p_p2 = stats.pearsonr(nz['freq_mhz'], nz['average_waittime_ns'])
        r_s2, p_s2 = stats.spearmanr(nz['freq_mhz'], nz['average_waittime_ns'])
        print(f"\nCorrelation (nonzero waittime rows only, n={len(nz):,}):")
        print(f"  Pearson  r = {r_p2:+.4f}  p = {p_p2:.2e}")
        print(f"  Spearman r = {r_s2:+.4f}  p = {p_s2:.2e}")

    # --- Mean/median waittime per frequency ---
    print("\nWaittime summary by freq_mhz:")
    summary = df.groupby('freq_mhz')['average_waittime_ns'].agg(
        mean='mean', median='median', count='count'
    ).round(1)
    print(summary.to_string())

    # --- Per-op-type breakdown ---
    df2 = df.copy()
    df2['op_type'] = df2['op_name'].str.replace(r'\(.*', '', regex=True)
    top_ops = df2['op_type'].value_counts().head(8).index
    print(f"\nPearson r (waittime vs freq) per op type (top {len(top_ops)} by count):")
    print(f"  {'op_type':<14} {'n':>8}  {'r':>8}  {'p':>10}  significance")
    for op in top_ops:
        sub = df2[df2['op_type'] == op]
        if sub['freq_mhz'].nunique() < 2:
            continue
        r, p = stats.pearsonr(sub['freq_mhz'], sub['average_waittime_ns'])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {op:<14} {len(sub):>8,}  {r:>+8.4f}  {p:>10.2e}  {sig}")

# --- Plot: mean waittime vs freq for both benchmarks ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, df) in zip(axes, [("LU", lu), ("Cholesky", ch)]):
    grouped = df.groupby('freq_mhz')['average_waittime_ns'].mean()
    ax.bar(grouped.index.astype(str), grouped.values, color='steelblue', edgecolor='black')
    ax.set_title(f"{name}: Mean waittime vs Frequency")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Mean waittime (ns)")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("waittime_freq_correlation.png", dpi=150)
print("\nPlot saved to waittime_freq_correlation.png")

# --- Relative execution time: local vs remote aggregated median ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, df) in zip(axes, [("LU", lu), ("Cholesky", ch)]):
    baseline_freq = df['freq_mhz'].min()
    df2 = df.copy()
    df2['locality'] = df2['op_name'].apply(classify_locality)

    for locality, color in [('local', 'steelblue'), ('remote', 'tomato')]:
        sub = df2[df2['locality'] == locality]
        if sub.empty:
            continue
        median_by_freq = sub.groupby('freq_mhz')['executiontime_ns'].median()
        baseline = median_by_freq.get(baseline_freq)
        if pd.isna(baseline) or baseline == 0:
            print(f"  WARNING [{name}] {locality}: no data at baseline freq")
            continue
        relative = median_by_freq / baseline
        ax.plot(relative.index, relative.values, marker='o', markersize=4, linewidth=1.5,
                color=color, label=locality)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f"{name}: Relative median execution time — local vs remote")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Relative execution time (1.0 = 240 MHz baseline)")
    ax.legend()

plt.tight_layout()
plt.savefig("local_vs_remote_exectime.png", dpi=150)
print("Plot saved to local_vs_remote_exectime.png")

# --- CSV: median execution time at every frequency per unique op ---
for name, df in [("LU", lu), ("Cholesky", ch)]:
    freqs = sorted(df['freq_mhz'].unique())
    baseline_freq = freqs[0]
    rows = []
    for op_name, sub in df.groupby('op_name'):
        op_type = op_name.split('(')[0]
        locality = classify_locality(op_name)
        median_by_freq = sub.groupby('freq_mhz')['executiontime_ns'].median()
        baseline = median_by_freq.get(baseline_freq)
        if pd.isna(baseline) or baseline == 0:
            print(f"  WARNING [{name}] {op_name}: no data at {baseline_freq} MHz — skipping")
            continue
        row = {'op_name': op_name, 'op_type': op_type, 'locality': locality}
        for freq in freqs:
            med = median_by_freq.get(freq)
            if pd.isna(med):
                print(f"  WARNING [{name}] {op_name}: no data at {freq} MHz")
            row[f'median_ns_{freq}'] = med
            row[f'relative_{freq}'] = med / baseline if not pd.isna(med) else None
        row['pct_change_low_to_high'] = 100 * (median_by_freq.get(freqs[-1]) - baseline) / baseline
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = f"{name.lower()}_exectime_by_op.csv"
    out.to_csv(out_path, index=False)
    unknown = (out['locality'] == 'unknown').sum()
    if unknown:
        print(f"  WARNING [{name}]: {unknown} ops with unknown locality — check classify_locality()")
    print(f"\nSaved {out_path} ({len(out)} tasks, {len(freqs)} frequencies, locality classified)")
