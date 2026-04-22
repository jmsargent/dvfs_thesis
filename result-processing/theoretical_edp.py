import re
import io
import csv
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = "/data/users/sargent/dvfs_thesis/jobs/084-profile-runtimes-per-frequency-improved/slurm-174558.out"
DATA_DIR   = Path(__file__).parent / "data"

BASELINE_FREQ_HIGH = 2040
BASELINE_FREQ_LOW  = 240

METRICS = ["edp", "e2dp", "e3dp", "ed2p", "ed3p", "ed10p"]

START_EXP_RE = re.compile(r"START_EXPERIMENT: BENCH=(\w+)_FREQ=(\d+)_TILE=(\d+)_N=(\d+)_R=(\d+)")
TIMING_RE    = re.compile(r"(\w+ tiledStatic) (start|end)_time: ([\d.]+)")
CSV_BEGIN_RE = re.compile(r"--- BEGIN CSV_DATA\|(\w+)\|(\d+)\|(\d+)\|(\d+) ---")
CSV_END_RE   = re.compile(r"--- END CSV_DATA\|(\w+)\|(\d+)\|(\d+)\|(\d+) ---")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_runs(path) -> pd.DataFrame:
    """Parse kernel timing data → one row per individual run."""
    records = []
    current_exp = None
    pending_start = None
    run_idx = 0

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = START_EXP_RE.search(line)
            if m:
                current_exp = {
                    "bench":    m.group(1),
                    "freq_mhz": int(m.group(2)),
                    "tile":     int(m.group(3)),
                    "N":        int(m.group(4)),
                }
                pending_start = None
                run_idx = 0
                continue

            if current_exp is None:
                continue

            m = TIMING_RE.search(line)
            if m:
                ts = float(m.group(3))
                if m.group(2) == "start":
                    pending_start = ts
                elif m.group(2) == "end" and pending_start is not None:
                    records.append({
                        **current_exp,
                        "run_idx":    run_idx,
                        "t_start":    pending_start,
                        "t_end":      ts,
                        "duration_s": ts - pending_start,
                    })
                    pending_start = None
                    run_idx += 1

    return pd.DataFrame(records)


def parse_energy(path) -> pd.DataFrame:
    """Parse GPU power-monitor CSV blocks → one row per sample per GPU."""
    records = []
    in_csv = False
    csv_key = None
    csv_lines = []

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = CSV_BEGIN_RE.search(line)
            if m:
                in_csv = True
                csv_key = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                csv_lines = []
                continue

            m = CSV_END_RE.search(line)
            if m and in_csv:
                in_csv = False
                bench, freq, tile, gpu_id = csv_key
                reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
                for row in reader:
                    try:
                        records.append({
                            "bench":            bench,
                            "freq_mhz":         freq,
                            "tile":             tile,
                            "gpu_id":           gpu_id,
                            "timestamp":        float(row["timestamp"]),
                            "total_energy_mj":  float(row["total_energy_mj"]),
                        })
                    except (ValueError, KeyError):
                        pass
                csv_lines = []
                continue

            if in_csv:
                csv_lines.append(line)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Build results DataFrame
# ---------------------------------------------------------------------------

def energy_in_window(gpu_df: pd.DataFrame, t_start: float, t_end: float) -> float:
    """Return energy (mJ) consumed in [t_start, t_end] from cumulative readings."""
    pts = gpu_df[(gpu_df["timestamp"] >= t_start) & (gpu_df["timestamp"] <= t_end)]
    if len(pts) < 2:
        pts = gpu_df[(gpu_df["timestamp"] >= t_start - 1) & (gpu_df["timestamp"] <= t_end + 1)]
    if len(pts) < 2:
        return np.nan
    return pts["total_energy_mj"].iloc[-1] - pts["total_energy_mj"].iloc[0]


def build_results_df(runs_df: pd.DataFrame, energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate runs_df to one row per (bench, freq_mhz).
    Energy is measured on the run whose duration is closest to the median.
    """
    records = []
    for (bench, freq), group in runs_df.groupby(["bench", "freq_mhz"], sort=False):
        median_time = group["duration_s"].median()
        mean_time   = group["duration_s"].mean()
        std_time    = group["duration_s"].std()

        closest = group.loc[(group["duration_s"] - median_time).abs().idxmin()]

        bench_freq_energy = energy_df[(energy_df["bench"] == bench) & (energy_df["freq_mhz"] == freq)]
        n_gpus = bench_freq_energy["gpu_id"].nunique()
        total_energy_mj = sum(
            energy_in_window(
                bench_freq_energy[bench_freq_energy["gpu_id"] == gpu_id],
                closest["t_start"], closest["t_end"],
            )
            for gpu_id in bench_freq_energy["gpu_id"].unique()
        )

        energy_j = total_energy_mj / 1000

        def edp(e_exp, d_exp):
            return (energy_j ** e_exp) * (median_time ** d_exp)

        records.append({
            "bench":           bench,
            "freq_mhz":        freq,
            "median_time_s":   median_time,
            "mean_time_s":     mean_time,
            "std_time_s":      std_time,
            "n_runs":          len(group),
            "total_energy_mj": total_energy_mj,
            "n_gpus":          n_gpus,
            "edp":   edp(1, 1),
            "e2dp":  edp(2, 1),
            "e3dp":  edp(3, 1),
            "ed2p":  edp(1, 2),
            "ed3p":  edp(1, 3),
            "ed10p": edp(1, 10),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_raw_results(df: pd.DataFrame):
    cols = ["bench", "freq_mhz", "median_time_s", "mean_time_s", "std_time_s", "total_energy_mj", "edp"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nParsed {len(df)} experiments.")


def print_runtime_by_frequency(df: pd.DataFrame):
    print()
    print(f"{'Benchmark':<25} {'Freq (MHz)':>12} {'Median (s)':>12} {'Mean (s)':>12} {'Std (s)':>10}")
    print("-" * 75)
    for bench, group in df.groupby("bench"):
        for _, row in group.sort_values("freq_mhz").iterrows():
            print(f"{bench:<25} {int(row['freq_mhz']):>12} {row['median_time_s']:>12.4f} "
                  f"{row['mean_time_s']:>12.4f} {row['std_time_s']:>10.4f}")
        print()


def print_best_runtime_vs_max_freq(df: pd.DataFrame):
    print()
    print(f"{'Benchmark':<25} {'Best freq (MHz)':>16} {'Best (s)':>10} {'2040MHz (s)':>12} {'Speedup':>10}")
    print("-" * 78)
    for bench, group in df.groupby("bench"):
        best = group.loc[group["median_time_s"].idxmin()]
        best_freq = int(best["freq_mhz"])
        if best_freq == BASELINE_FREQ_HIGH:
            continue
        high_row = group[group["freq_mhz"] == BASELINE_FREQ_HIGH]
        if high_row.empty:
            continue
        t_high = high_row.iloc[0]["median_time_s"]
        speedup = t_high / best["median_time_s"]
        print(f"{bench:<25} {best_freq:>16} {best['median_time_s']:>10.4f} "
              f"{t_high:>12.4f} {speedup:>9.3f}x")


def print_optimal_frequencies(df: pd.DataFrame):
    print()
    print(f"{'Benchmark':<25} {'Metric':<8} {'Best freq (MHz)':>16} {'Value':>14} {'vs 2040MHz':>12} {'vs 240MHz':>12}")
    print("-" * 90)

    baselines = (
        df[df["freq_mhz"].isin([BASELINE_FREQ_HIGH, BASELINE_FREQ_LOW])]
        .set_index(["bench", "freq_mhz"])
    )

    for bench, group in df.groupby("bench"):
        for metric in METRICS:
            valid = group.dropna(subset=[metric])
            if valid.empty:
                continue

            best = valid.loc[valid[metric].idxmin()]

            try:
                val_high = baselines.loc[(bench, BASELINE_FREQ_HIGH), metric]
                val_low  = baselines.loc[(bench, BASELINE_FREQ_LOW),  metric]
                pct_high_str = f"{val_high / best[metric] * 100:.1f}%"
                pct_low_str  = f"{val_low  / best[metric] * 100:.1f}%"
            except KeyError:
                pct_high_str = pct_low_str = "n/a"

            print(f"{bench:<25} {metric:<8} {int(best['freq_mhz']):>16} {best[metric]:>14.4f} "
                  f"{pct_high_str:>12} {pct_low_str:>12}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stem = Path(INPUT_FILE).stem
    DATA_DIR.mkdir(exist_ok=True)

    runs_df   = parse_runs(INPUT_FILE)
    energy_df = parse_energy(INPUT_FILE)

    runs_df.to_csv(DATA_DIR / f"{stem}_runs.csv", index=False)
    energy_df.to_csv(DATA_DIR / f"{stem}_energy.csv", index=False)
    print(f"Saved {len(runs_df)} run rows  → {DATA_DIR}/{stem}_runs.csv")
    print(f"Saved {len(energy_df)} energy rows → {DATA_DIR}/{stem}_energy.csv")

    results_df = build_results_df(runs_df, energy_df)

    print_raw_results(results_df)
    print_runtime_by_frequency(results_df)
    print_best_runtime_vs_max_freq(results_df)
    print_optimal_frequencies(results_df)
