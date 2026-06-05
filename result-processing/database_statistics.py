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
    op_type = op_name.split("(")[0]
    params_str = op_name[len(op_type) + 1 :].rstrip(")")
    params = [int(p) for p in params_str.split(";")] if params_str else []

    always_local = {"POTRF", "GETRF", "TRSM", "TRSM_R"}
    if op_type in always_local:
        return "local"

    if op_type == "SYRK":
        return "local" if params[0] % N_PES == params[1] % N_PES else "remote"

    if op_type == "GEMM":
        return "local" if params[1] % N_PES == params[2] % N_PES else "remote"

    if op_type == "TRSM_L":
        return "local" if params[1] % N_PES == params[0] % N_PES else "remote"

    return "unknown"


def waittime_correlation(lu, ch):
    for name, df in [("LU", lu), ("Cholesky", ch)]:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(f"Rows: {len(df):,}  |  freq_mhz unique: {sorted(df['freq_mhz'].unique())}")
        nonzero_pct = 100 * df["average_waittime_ns"].gt(0).mean()
        print(
            f"average_waittime_ns: mean={df['average_waittime_ns'].mean():.1f}, "
            f"nonzero={df['average_waittime_ns'].gt(0).sum():,} ({nonzero_pct:.1f}%)"
        )

        r_p, p_p = stats.pearsonr(df["freq_mhz"], df["average_waittime_ns"])
        r_s, p_s = stats.spearmanr(df["freq_mhz"], df["average_waittime_ns"])
        print("\nOverall correlation (all rows):")
        print(f"  Pearson  r = {r_p:+.4f}  p = {p_p:.2e}")
        print(f"  Spearman r = {r_s:+.4f}  p = {p_s:.2e}")

        nz = df[df["average_waittime_ns"] > 0]
        if len(nz) > 1 and nz["freq_mhz"].nunique() > 1:
            r_p2, p_p2 = stats.pearsonr(nz["freq_mhz"], nz["average_waittime_ns"])
            r_s2, p_s2 = stats.spearmanr(nz["freq_mhz"], nz["average_waittime_ns"])
            print(f"\nCorrelation (nonzero waittime rows only, n={len(nz):,}):")
            print(f"  Pearson  r = {r_p2:+.4f}  p = {p_p2:.2e}")
            print(f"  Spearman r = {r_s2:+.4f}  p = {p_s2:.2e}")

        print("\nWaittime summary by freq_mhz:")
        summary = (
            df.groupby("freq_mhz")["average_waittime_ns"]
            .agg(mean="mean", median="median", count="count")
            .round(1)
        )
        print(summary.to_string())

        df2 = df.copy()
        df2["op_type"] = df2["op_name"].str.replace(r"\(.*", "", regex=True)
        top_ops = df2["op_type"].value_counts().head(8).index
        print(f"\nPearson r (waittime vs freq) per op type (top {len(top_ops)} by count):")
        print(f"  {'op_type':<14} {'n':>8}  {'r':>8}  {'p':>10}  significance")
        for op in top_ops:
            sub = df2[df2["op_type"] == op]
            if sub["freq_mhz"].nunique() < 2:
                continue
            r, p = stats.pearsonr(sub["freq_mhz"], sub["average_waittime_ns"])
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"  {op:<14} {len(sub):>8,}  {r:>+8.4f}  {p:>10.2e}  {sig}")


def plot_waittime_vs_freq(lu, ch):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, df) in zip(axes, [("LU", lu), ("Cholesky", ch)]):
        grouped = df.groupby("freq_mhz")["average_waittime_ns"].mean()
        ax.bar(
            grouped.index.astype(str), grouped.values, color="steelblue", edgecolor="black"
        )
        ax.set_title(f"{name}: Mean waittime vs Frequency")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Mean waittime (ns)")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("waittime_freq_correlation.png", dpi=150)
    print("Plot saved to waittime_freq_correlation.png")


def plot_local_vs_remote(lu, ch):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, df) in zip(axes, [("LU", lu), ("Cholesky", ch)]):
        baseline_freq = df["freq_mhz"].min()
        df2 = df.copy()
        df2["locality"] = df2["op_name"].apply(classify_locality)

        for locality, color in [("local", "steelblue"), ("remote", "tomato")]:
            sub = df2[df2["locality"] == locality]
            if sub.empty:
                continue
            median_by_freq = sub.groupby("freq_mhz")["executiontime_ns"].median()
            baseline = median_by_freq.get(baseline_freq)
            if pd.isna(baseline) or baseline == 0:
                print(f"  WARNING [{name}] {locality}: no data at baseline freq")
                continue
            relative = median_by_freq / baseline
            ax.plot(
                relative.index,
                relative.values,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label=locality,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{name}: Relative median execution time — local vs remote")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Relative execution time (1.0 = 240 MHz baseline)")
        ax.legend()

    plt.tight_layout()
    plt.savefig("local_vs_remote_exectime.png", dpi=150)
    print("Plot saved to local_vs_remote_exectime.png")


def export_exectime_csv(lu, ch):
    for name, df in [("LU", lu), ("Cholesky", ch)]:
        freqs = sorted(df["freq_mhz"].unique())
        baseline_freq = freqs[0]
        rows = []
        for op_name, sub in df.groupby("op_name"):
            op_type = op_name.split("(")[0]
            locality = classify_locality(op_name)
            median_by_freq = sub.groupby("freq_mhz")["executiontime_ns"].median()
            baseline = median_by_freq.get(baseline_freq)
            if pd.isna(baseline) or baseline == 0:
                print(f"  WARNING [{name}] {op_name}: no data at {baseline_freq} MHz — skipping")
                continue
            row = {"op_name": op_name, "op_type": op_type, "locality": locality}
            for freq in freqs:
                med = median_by_freq.get(freq)
                if pd.isna(med):
                    print(f"  WARNING [{name}] {op_name}: no data at {freq} MHz")
                row[f"median_ns_{freq}"] = med
                row[f"relative_{freq}"] = med / baseline if not pd.isna(med) else None
            row["pct_change_low_to_high"] = (
                100 * (median_by_freq.get(freqs[-1]) - baseline) / baseline
            )
            rows.append(row)

        out = pd.DataFrame(rows)
        out_path = f"{name.lower()}_exectime_by_op.csv"
        out.to_csv(out_path, index=False)
        unknown = (out["locality"] == "unknown").sum()
        if unknown:
            print(f"  WARNING [{name}]: {unknown} ops with unknown locality — check classify_locality()")
        print(f"\nSaved {out_path} ({len(out)} tasks, {len(freqs)} frequencies, locality classified)")


def kernel_median_and_cv(lu, ch):
    print("\nMedian execution time and coefficient of variation per kernel:")
    for name, df in [("LU", lu), ("Cholesky", ch)]:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        df2 = df[df["freq_mhz"] == 2040].copy()
        # df2 = df[df["freq_mhz"] == 240].copy()
        df2["op_type"] = df2["op_name"].str.replace(r"\(.*", "", regex=True)
        df2["locality"] = df2["op_name"].apply(classify_locality)
        stats_by_kernel = df2.groupby(["op_type", "locality"])["executiontime_ns"].agg(
            median_s=lambda x: x.median() / 1e9,
            cv=lambda x: x.std() / x.mean() if x.mean() != 0 else float("nan"),
            count="count",
        )
        print(f"  {'kernel':<14} {'locality':<10} {'median (s)':>14}  {'cv':>8}  {'n':>8}")
        print(f"  {'-' * 14} {'-' * 10} {'-' * 14}  {'-' * 8}  {'-' * 8}")
        for (op_type, locality), row in stats_by_kernel.iterrows():
            print(f"  {op_type:<14} {locality:<10} {row['median_s']:>14.6f}  {row['cv']:>8.4f}  {row['count']:>8,}")


SECTIONS = {
    "1": ("Waittime correlation stats", waittime_correlation),
    "2": ("Plot: mean waittime vs freq", plot_waittime_vs_freq),
    "3": ("Plot: local vs remote execution time", plot_local_vs_remote),
    "4": ("Export per-op execution time CSV", export_exectime_csv),
    "5": ("Kernel median execution time and CV", kernel_median_and_cv),
}


def main():
    db = "/Users/jonathansargent/dvfs_thesis/db_new_settings"
    lu = pd.read_csv(f"{db}/lu_database.csv")
    ch = pd.read_csv(f"{db}/cholesky_database.csv")

    print("Available sections:")
    for key, (label, _) in SECTIONS.items():
        print(f"  {key}: {label}")
    print("  a: run all")

    choice = input("\nRun which section(s)? (e.g. '1', '1 3', 'a'): ").strip().lower()

    if choice == "a":
        keys = list(SECTIONS.keys())
    else:
        keys = choice.split()

    for key in keys:
        if key not in SECTIONS:
            print(f"Unknown section '{key}', skipping.")
            continue
        label, fn = SECTIONS[key]
        print(f"\n{'#' * 60}")
        print(f"  {label}")
        print(f"{'#' * 60}")
        fn(lu, ch)


if __name__ == "__main__":
    main()
