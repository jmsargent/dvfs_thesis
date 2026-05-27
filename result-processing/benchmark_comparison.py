import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/Users/jonathansargent/dvfs_thesis/experiments"

SWEEP = "freq-sweep"
TUNER = "constant"

_PROFILE_PATTERN = re.compile(r"tasks_pe(\d+)\.csv")
_ENERGY_PATTERN = re.compile(r"gpu_(\d+)\.csv")

BENCHMARK = "cholesky"

GOAL_FUNCS = {
    "edp": lambda e, d: e * d,
    "ed2p": lambda e, d: e * d**2,
    "e2dp": lambda e, d: e**2 * d,
    "energy": lambda e, _: e,
}

def get_available_freqs(experiment_path: str) -> list[int]:
    return sorted(int(d) for d in os.listdir(experiment_path) if d.isdigit())

def _scan_pe_csvs(directory: str, pattern: re.Pattern) -> dict[int, pd.DataFrame]:
    """Scan directory, return {pe: DataFrame} for every filename matching pattern."""
    result = {}
    for fname in sorted(os.listdir(directory)):
        m = pattern.fullmatch(fname)
        if m:
            result[int(m.group(1))] = pd.read_csv(os.path.join(directory, fname))
    return result

def get_profile_dfs(experiment_path: str) -> pd.DataFrame:
    """Return {pe: DataFrame} for task/profile CSVs under experiment_path/profile/."""
    dfs = _scan_pe_csvs(os.path.join(experiment_path, "profile"), _PROFILE_PATTERN)
    return pd.concat(dfs.values(), ignore_index=True)


def get_energy_dfs(experiment_path: str) -> pd.DataFrame:
    """Return concatenated energy DataFrame with a pe column derived from the filename."""
    dfs = _scan_pe_csvs(os.path.join(experiment_path, "energy"), _ENERGY_PATTERN)
    frames = []
    for pe, df in dfs.items():
        df.insert(0, "pe", pe)
        df["timestamp_ns"] = (df["timestamp"] * 1e9).astype("int64")
        df["total_energy_j"] = df["total_energy_mj"] / 1000
        df = df.drop(columns=["timestamp", "datetime", "total_energy_mj"])
        frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_ns")
        .reset_index(drop=True)
    )

def energy_per_run_df(run_bounds, energy_dfs):
    rows = []
    for pe, pe_energy in energy_dfs.groupby("pe"):
        ts = pe_energy["timestamp_ns"].values
        energy = pe_energy["total_energy_j"].values
        for _, run_row in run_bounds.iterrows():
            idx_start = np.searchsorted(ts, run_row["ts_start_ns"], side="left")
            idx_end = np.searchsorted(ts, run_row["ts_end_ns"], side="right") - 1
            rows.append(
                {
                    "pe": pe,
                    "run": int(run_row["run"]),
                    "total_energy_start_j": energy[idx_start]
                    if idx_start < len(energy)
                    else float("nan"),
                    "total_energy_end_j": energy[idx_end]
                    if idx_end >= 0
                    else float("nan"),
                }
            )

    energy_per_run_per_pe = pd.DataFrame(rows)
    energy_per_run_per_pe["total_energy_j"] = (
        energy_per_run_per_pe["total_energy_end_j"]
        - energy_per_run_per_pe["total_energy_start_j"]
    )
    energy_per_run = (
        energy_per_run_per_pe.groupby("run")["total_energy_j"].sum().reset_index()
    )
    return energy_per_run


def runtimes_dfs(profile_dfs_sweep):
    run_bounds = (
        profile_dfs_sweep.groupby("run")
        .agg(ts_start_ns=("start_ts", "min"), ts_end_ns=("end_ts", "max"))
        .reset_index()
    )
    run_bounds["execution_time_s"] = (
        run_bounds["ts_end_ns"] - run_bounds["ts_start_ns"]
    ) / 1e9
    execution_times = run_bounds[["run", "execution_time_s"]]
    return run_bounds, execution_times


def frequency_stats_row(f, execution_times, energy_per_run):
    mean_exec_s = execution_times["execution_time_s"].mean()
    mean_energy_j = energy_per_run["total_energy_j"].mean()
    return {
        "frequency": f,
        "mean_exec_s": mean_exec_s,
        "mean_energy_j": mean_energy_j,
        "median_exec_s": execution_times["execution_time_s"].median(),
        "median_energy_j": energy_per_run["total_energy_j"].median(),
        "energy_rsd": energy_per_run["total_energy_j"].std() / mean_energy_j * 100,
        "exec_rsd": execution_times["execution_time_s"].std() / mean_exec_s * 100,
        "edp": GOAL_FUNCS["edp"](mean_energy_j, mean_exec_s),
        "ed2p": GOAL_FUNCS["ed2p"](mean_energy_j, mean_exec_s),
        "e2dp": GOAL_FUNCS["e2dp"](mean_energy_j, mean_exec_s),
    }


def goals(baseline_df):
    metrics = ["edp", "ed2p", "e2dp", "mean_energy_j", "mean_exec_s"]
    return {
        m: (baseline_df.loc[baseline_df[m].idxmin(), "frequency"], baseline_df[m].min())
        for m in metrics
    }

def execution_times(df):
    return df[["frequency", "mean_exec_s", "median_exec_s", "cv_exec_percent"]]
    

def dvfs_algo_stats(algo, tuner, benchmark):

    base_path = f"{BASE}/{algo}/{benchmark}/{tuner}"
    frequencies = get_available_freqs(base_path)

    rows = []
    for f in frequencies:
        inner_path = f"{base_path}/{f}"

        profile_dfs_sweep = get_profile_dfs(inner_path)
        run_bounds, execution_times = runtimes_dfs(profile_dfs_sweep)

        energy_dfs_sweep = get_energy_dfs(inner_path)
        energy_per_run = energy_per_run_df(run_bounds, energy_dfs_sweep)

        next_row = frequency_stats_row(f, execution_times, energy_per_run)
        rows.append(next_row)

    return pd.DataFrame(rows)


def compare_algos(stats_baseline_df, stats_other_tuple):
    (other_name, stats_other_df) = stats_other_tuple

    baseline_results = goals(stats_baseline_df)
    other_results = goals(stats_other_df)

    rows_compare = []
    rows_freq = []
    
    for metric, _ in baseline_results.items():
        b_freq,b_result = baseline_results[metric]
        o_freq,o_result = other_results[metric]
        
        ratio = b_result / o_result
        
        rows_compare.append(
            {
                "metric": metric,
                "baseline": b_result,
                other_name: o_result,
                "ratio": ratio
            }
        )
        
        rows_freq.append(
            {
                "metric": metric,
                "baseline": b_freq,
                other_name: o_freq,
            }
        )
        
        
    return pd.DataFrame(rows_compare), pd.DataFrame(rows_freq)

def plots():
    fig, ax = plt.subplots(figsize=(8, 5))
    for benchmark in ("lu", "cholesky"):
        df = dvfs_algo_stats(SWEEP, TUNER, benchmark)
        ax.plot(df["frequency"], df["mean_exec_s"], marker="o", label=benchmark)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Mean execution time (s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":


    for bench in ("lu", "cholesky"):
        baseline_df = dvfs_algo_stats(SWEEP, TUNER, bench)
        baseline_scores = goals(baseline_df)
        goal_labels = {
            "edp": "$EDP$",
            "ed2p": "$ED^{2}P$",
            "e2dp": "$E^{2}DP$",
            "mean_energy_j": "{Mean Energy (J)}",
            "mean_exec_s": "{Mean Execution Time (s)}",
        }
        def fmt_score(val):
            mantissa, exp = f"{val:.3e}".split("e")
            return f"{{${mantissa} \\times 10^{{{int(exp)}}}$}}"

        rows = [
            {"goal": goal_labels[k], "best_frequency": v[0], "score": fmt_score(v[1])}
            for k, v in baseline_scores.items()
        ]
        out_df = pd.DataFrame(rows, columns=["goal", "best_frequency", "score"])
        out_path = f"baseline-csv/{bench}_baseline_goals.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        print(out_df.to_string(index=False))
    
    
    # I wanna check for what time is the program downtunes vs what time is it at base frequency
    # 
    
    # print(baseline_df)
    
    
    # other_folder = "slackaware"
    # stats = dvfs_algo_stats(other_folder, "combined-slack", BENCHMARK)
    # df_compare, df_freqs = compare_algos(baseline_df, (other_folder, stats))

    # # print(baseline_df)
    # # print(baseline_df)
    # print(stats)
    # stats.to_csv(f"{other_folder}-csv/{BENCHMARK}_performance_low_frequency_sweep.csv", index=False)
    # print("comparison:")
    # print(df_compare)
    # metric_labels = {
    #     "edp": "EDP",
    #     "ed2p": r"ED$^2$P",
    #     "e2dp": r"E$^2$DP",
    #     "mean_energy_j": "Mean Energy (J)",
    #     "mean_exec_s": "Mean Time (s)",
    # }
    # df_compare["metric"] = df_compare["metric"].replace(metric_labels)
    # df_freqs["metric"] = df_freqs["metric"].replace(metric_labels)
    # df_compare = df_compare.rename(columns={other_folder: "Slack-Aware"})
    # df_freqs = df_freqs.rename(columns={other_folder: "Slack-Aware"})

    # df_compare.to_csv(f"{other_folder}-csv/{BENCHMARK}_baseline_goal_comparison.csv", index=False)
    # # print("freqs:")
    # print(df_freqs)
    # df_freqs.to_csv(f"{other_folder}-csv/{BENCHMARK}_optimal_freq_per_goal_comparison.csv", index=False)

    # # print("=====")
    # # print(execution_times(baseline_df))