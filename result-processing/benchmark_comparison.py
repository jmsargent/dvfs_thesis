import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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


def single_freq_stats(algo, tuner, benchmark, freq) -> pd.DataFrame:
    """Return a one-row DataFrame with stats for a single fixed frequency."""
    inner_path = f"{BASE}/{algo}/{benchmark}/{tuner}/{freq}"
    profile_dfs = get_profile_dfs(inner_path)
    run_bounds, exec_times = runtimes_dfs(profile_dfs)
    energy_dfs = get_energy_dfs(inner_path)
    energy_per_run = energy_per_run_df(run_bounds, energy_dfs)
    return pd.DataFrame([frequency_stats_row(freq, exec_times, energy_per_run)])


def compare_algos(stats_baseline_df, *others):
    baseline_results = goals(stats_baseline_df)
    others_results = [(name, goals(df)) for name, df in others]

    rows_compare = []
    rows_freq = []

    for metric in baseline_results:
        b_freq, b_result = baseline_results[metric]

        compare_row = {"metric": metric, "baseline": b_result}
        freq_row = {"metric": metric, "baseline": b_freq}

        for name, other_results in others_results:
            o_freq, o_result = other_results[metric]
            compare_row[name] = o_result
            compare_row[f"{name}_ratio"] = b_result / o_result
            freq_row[name] = o_freq

        rows_compare.append(compare_row)
        rows_freq.append(freq_row)

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
    
def welchs_ttest(times_per_freq: dict, alpha=0.05):
    """Given {freq: execution_times_array} (sorted), return the highest frequency
    where stepping up still yields a statistically significant speedup."""
    freqs = sorted(times_per_freq)
    corrected_alpha = alpha / (len(freqs) - 1)
    saturation_freq = freqs[0]
    for f_lo, f_hi in zip(freqs, freqs[1:]):
        _, p = stats.ttest_ind(times_per_freq[f_lo], times_per_freq[f_hi],
                               equal_var=False, alternative="greater")
        if p < corrected_alpha:
            saturation_freq = f_hi
    return saturation_freq

if __name__ == "__main__":

    for b in ["lu", "cholesky"]:
        df_2040 = single_freq_stats("saturate-functional-units", TUNER, b, 2040)
        df_saturate = dvfs_algo_stats("saturate-functional-units", TUNER, b)
        df_syncpoints = dvfs_algo_stats("syncpoints-bk-sweep", "combined-slack", b)
        df_slackaware_sweep = dvfs_algo_stats("slackaware-bk-sweep", "combined-slack", b)
        compare_df, freq_df = compare_algos(
            df_2040,
            ("constant-sweep", df_saturate),
            ("syncpoints", df_syncpoints),
            ("slackaware-sweep", df_slackaware_sweep),
        )
        print(f"\n{b}\n", compare_df.to_string(index=False))
        print(freq_df.to_string(index=False))

    # fig, ax = plt.subplots(figsize=(8, 5))
    # for b in ["lu", "cholesky"]:
    #     print(f"\n--- {b} ---")
    #     base_path = f"{BASE}/sweep-big-kernels2/{b}/constant"
    #     freqs = get_available_freqs(base_path)
    #     times_per_freq = {f: runtimes_dfs(get_profile_dfs(f"{base_path}/{f}"))[1]["execution_time_s"].values for f in freqs}
    #     df = dvfs_algo_stats('sweep-big-kernels2', 'constant', b)
    #     for g, (freq, val) in goals(df).items():
    #         print(f"  {g}: freq={freq}, value={val}")
    #     print(f"  saturation_freq: {welchs_ttest(times_per_freq)}")
    #     ax.plot(df["frequency"], df["mean_exec_s"], marker="o", label=b)

    # ax.set_xlabel("Frequency (MHz)")
    # ax.set_ylabel("Mean Execution Time (s)")
    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)
    # ax.set_xticks(freqs)
    # ax.tick_params(axis="x", rotation=45)
    # ax.legend()
    # ax.grid(True, linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()

# 1365 cholesky
# 1440 lu