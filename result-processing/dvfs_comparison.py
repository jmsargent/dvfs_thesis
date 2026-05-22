import argparse

import pandas as pd
import numpy as np
import os

BASE = "/Users/jonathansargent/dvfs_thesis/experiments"

HOMO_BASE = f"{BASE}/homogenous-retune-best-parameters"
SYNC_BASE = f"{BASE}/retune_on_sync2"
SLACKAWARE_BASE = f"{BASE}/slackaware3"
IMPROVED_SLACKAWARE = f"{BASE}/improved-slackawareness"
IMPROVED_SLACKAWARE_POST_BUGFIX = f"{BASE}/improved-slackawareness-post-bugfix"

BENCHMARKS = ["cholesky", "lu"]
PES = [0, 1, 2, 3]

GOAL_FUNCS = {
    "edp":    lambda e, d: e * d,
    "ed2p":   lambda e, d: e * d**2,
    "e2dp":   lambda e, d: e**2 * d,
    "energy": lambda e, _: e,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_energy_func(df_energy):
    """Linear interpolant: f(t) -> cumulative energy (mJ) at unix time t."""
    ts = df_energy["timestamp"].values
    cumulative_energy = df_energy["total_energy_mj"].values
    return lambda t: np.interp(t, ts, cumulative_energy)


def load_profiles(base, pes, energy_name="gpu_{pe}.csv", profile_name="profile_pe{pe}.csv"):
    energy_funcs = [
        make_energy_func(pd.read_csv(f"{base}/energy/{energy_name.format(pe=pe)}"))
        for pe in pes
    ]
    df_profile = pd.concat(
        [pd.read_csv(f"{base}/profile/{profile_name.format(pe=pe)}") for pe in pes]
    )
    return energy_funcs, df_profile


def score_runs(energy_funcs, df_profile, goal_func):
    """Return mean/median/cv of goal_func scores across runs."""
    run_bounds = (
        df_profile.groupby("run").agg(
            first_task_start=("start_ts", "min"),
            last_task_end=("end_ts", "max"),
        )
        / 1e9
    )
    delay = run_bounds["last_task_end"] - run_bounds["first_task_start"]
    energy = run_bounds.apply(
        lambda row: sum(
            ef(row["last_task_end"]) - ef(row["first_task_start"]) for ef in energy_funcs
        ),
        axis=1,
    )
    scores = goal_func(energy, delay)
    return pd.Series({"mean": scores.mean(), "median": scores.median(), "cv": scores.std() / scores.mean()})


def score_all_metrics(energy_funcs, df_profile):
    return {
        name: score_runs(energy_funcs, df_profile, goal_func)["mean"]
        for name, goal_func in GOAL_FUNCS.items()
    }


def _list_int_subdirs(path):
    return sorted(int(d) for d in os.listdir(path) if d.isdigit())


def load_homo_df(benchmark):
    """Homogeneous freq sweep — DataFrame indexed by frequency (MHz)."""
    freqs = sorted(
        int(d.split("_")[1])
        for d in os.listdir(HOMO_BASE)
        if d.startswith(f"{benchmark}_")
    )
    rows = []
    for freq in freqs:
        energy_funcs, df_profile = load_profiles(f"{HOMO_BASE}/{benchmark}_{freq}", PES)
        rows.append({"freq": freq, **score_all_metrics(energy_funcs, df_profile)})
    return pd.DataFrame(rows).set_index("freq")


def load_sync_scores(benchmark):
    """Sync-retune scores — dict of metric_name -> mean score.

    Sync has per-metric experiment directories, so each metric is loaded
    from its own path.
    """
    scores = {}
    for name, goal_func in GOAL_FUNCS.items():
        energy_funcs, df_profile = load_profiles(
            f"{SYNC_BASE}/{benchmark}/{name}", PES, profile_name="tasks_pe{pe}.csv"
        )
        scores[name] = score_runs(energy_funcs, df_profile, goal_func)["mean"]
    return scores


def load_combined_slack_df(base, benchmark):
    """Combined-slack freq sweep — DataFrame indexed by frequency (MHz)."""
    freqs = _list_int_subdirs(f"{base}/{benchmark}/combined-slack")
    rows = []
    for freq in freqs:
        energy_funcs, df_profile = load_profiles(
            f"{base}/{benchmark}/combined-slack/{freq}",
            PES,
            profile_name="tasks_pe{pe}.csv",
        )
        rows.append({"freq": freq, **score_all_metrics(energy_funcs, df_profile)})
    return pd.DataFrame(rows).set_index("freq")


def best_scores(df):
    """Return dict of metric -> score at the frequency that minimises that metric."""
    return {name: df.loc[df[name].idxmin(), name] for name in GOAL_FUNCS}


def load_scores_df(exp_base, benchmark):
    """Return a DataFrame of metric scores indexed by frequency, auto-detecting layout."""
    combined_path = f"{exp_base}/{benchmark}/combined-slack"
    if os.path.isdir(combined_path):
        return load_combined_slack_df(exp_base, benchmark)

    homo_dirs = [d for d in os.listdir(exp_base) if d.startswith(f"{benchmark}_")]
    if homo_dirs:
        freqs = sorted(int(d.split("_")[1]) for d in homo_dirs)
        rows = []
        for freq in freqs:
            energy_funcs, df_profile = load_profiles(
                f"{exp_base}/{benchmark}_{freq}", PES
            )
            rows.append({"freq": freq, **score_all_metrics(energy_funcs, df_profile)})
        return pd.DataFrame(rows).set_index("freq")

    raise ValueError(f"Unrecognised experiment layout in {exp_base} for benchmark '{benchmark}'")


def _mean_run_time_from_tasks(profile_dir, profile_name="tasks_pe{pe}.csv"):
    """Mean wall-clock run time (s) derived from task start/end timestamps."""
    dfs = [
        pd.read_csv(path)
        for pe in PES
        if os.path.exists(path := f"{profile_dir}/{profile_name.format(pe=pe)}")
    ]
    if not dfs:
        return float("nan")
    run_bounds = pd.concat(dfs).groupby("run").agg(t0=("start_ts", "min"), t1=("end_ts", "max"))
    return ((run_bounds["t1"] - run_bounds["t0"]) / 1e9).mean()


def load_exec_times(exp_base, benchmark):
    """Return a Series of mean run time (s) indexed by frequency (MHz).

    Detects layout automatically:
      - combined-slack: {benchmark}/combined-slack/{freq}/profile/
      - homogeneous:    {benchmark}_{freq}/profile/
    """
    combined_path = f"{exp_base}/{benchmark}/combined-slack"
    if os.path.isdir(combined_path):
        freqs = _list_int_subdirs(combined_path)
        return pd.Series(
            {freq: _mean_run_time_from_tasks(f"{combined_path}/{freq}/profile") for freq in freqs},
            name=os.path.basename(exp_base),
        )

    homo_dirs = [d for d in os.listdir(exp_base) if d.startswith(f"{benchmark}_")]
    if homo_dirs:
        freqs = sorted(int(d.split("_")[1]) for d in homo_dirs)
        return pd.Series(
            {
                freq: _mean_run_time_from_tasks(
                    f"{exp_base}/{benchmark}_{freq}/profile",
                    profile_name="profile_pe{pe}.csv",
                )
                for freq in freqs
            },
            name=os.path.basename(exp_base),
        )

    raise ValueError(f"Unrecognised experiment layout in {exp_base} for benchmark '{benchmark}'")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(benchmark):
    df_homo = load_homo_df(benchmark)
    sync = load_sync_scores(benchmark)
    v1 = best_scores(load_combined_slack_df(SLACKAWARE_BASE, benchmark))
    v2 = best_scores(load_combined_slack_df(IMPROVED_SLACKAWARE, benchmark))
    v2_fix = best_scores(load_combined_slack_df(IMPROVED_SLACKAWARE_POST_BUGFIX, benchmark))

    header = (
        f"{'metric':<8}  {'slack_v1':>12}  {'slack_v2':>12}  {'v2_fix':>12}  "
        f"{'sync_retune':>12}  {'best_homo':>12}  {'homo_freq':>10}  "
        f"{'v1/sync':>8}  {'v2/sync':>8}  {'fix/sync':>9}  {'fix/v2':>8}"
    )
    print(f"\n=== {benchmark} ===")
    print(header)
    print("-" * len(header))
    for name in GOAL_FUNCS:
        best_homo_freq = df_homo[name].idxmin()
        best_homo = df_homo.loc[best_homo_freq, name]
        print(
            f"{name:<8}  {v1[name]:>12.3e}  {v2[name]:>12.3e}  {v2_fix[name]:>12.3e}  "
            f"{sync[name]:>12.3e}  {best_homo:>12.3e}  {str(best_homo_freq) + ' MHz':>10}  "
            f"{sync[name]/v1[name]:>7.2f}x  {sync[name]/v2[name]:>7.2f}x  "
            f"{sync[name]/v2_fix[name]:>8.2f}x  {v2[name]/v2_fix[name]:>7.2f}x"
        )


def print_freq_sweep(df, label, benchmark, freq=None):
    if freq is not None:
        df = df.loc[[freq]] if freq in df.index else df.iloc[0:0]

    best_at = {fq: [] for fq in df.index}
    for name in GOAL_FUNCS:
        if not df.empty:
            best_at[df[name].idxmin()].append(name)

    header = f"{'freq':>8}  " + "  ".join(f"{name:>12}" for name in GOAL_FUNCS) + "  best_at"
    print(f"\n=== {label} — {benchmark} (freq sweep) ===")
    print(header)
    print("-" * len(header))
    for fq, row in df.iterrows():
        metrics = ", ".join(best_at[fq]) if best_at[fq] else ""
        print(
            f"{str(fq) + ' MHz':>8}  "
            + "  ".join(f"{row[name]:>12.3e}" for name in GOAL_FUNCS)
            + f"  {metrics}"
        )


def print_best_vs_homo(benchmark):
    df_homo = load_homo_df(benchmark)
    df_v2_fix = load_combined_slack_df(IMPROVED_SLACKAWARE_POST_BUGFIX, benchmark)

    header = (
        f"{'metric':<8}  {'best_fix':>12}  {'fix_freq':>10}  "
        f"{'best_homo':>12}  {'homo_freq':>10}  {'homo/fix':>9}"
    )
    print(f"\n=== best v2_fix vs best homo — {benchmark} ===")
    print(header)
    print("-" * len(header))
    for name in GOAL_FUNCS:
        best_fix_freq = df_v2_fix[name].idxmin()
        best_fix = df_v2_fix.loc[best_fix_freq, name]
        best_homo_freq = df_homo[name].idxmin()
        best_homo = df_homo.loc[best_homo_freq, name]
        print(
            f"{name:<8}  {best_fix:>12.3e}  {str(best_fix_freq) + ' MHz':>10}  "
            f"{best_homo:>12.3e}  {str(best_homo_freq) + ' MHz':>10}  {best_homo/best_fix:>8.2f}x"
        )


def print_metric_scores(experiment_names, benchmarks, metrics, freq=None):
    for benchmark in benchmarks:
        series = {}
        for name in experiment_names:
            try:
                df = load_scores_df(f"{BASE}/{name}", benchmark)
                series[name] = df[metrics] if len(metrics) > 1 else df[metrics[0]]
            except Exception as exc:
                print(f"[skipped {name}: {exc}]")

        if not series:
            continue

        all_freqs = sorted(set().union(*[
            set(s.index if isinstance(s, (pd.DataFrame, pd.Series)) else []) for s in series.values()
        ]))
        if freq is not None:
            all_freqs = [freq] if freq in all_freqs else []

        col_w = 14
        if len(metrics) == 1:
            header = f"{'freq':>8}  " + "  ".join(f"{n:>{col_w}}" for n in series)
            print(f"\n=== {benchmark} — {metrics[0]} scores ===")
            print(header)
            print("-" * len(header))
            for fq in all_freqs:
                row = f"{str(fq) + ' MHz':>8}  "
                row += "  ".join(
                    f"{series[n].loc[fq]:>{col_w}.3e}" if fq in series[n].index else f"{'N/A':>{col_w}}"
                    for n in series
                )
                print(row)
        else:
            for metric in metrics:
                header = f"{'freq':>8}  " + "  ".join(f"{n:>{col_w}}" for n in series)
                print(f"\n=== {benchmark} — {metric} scores ===")
                print(header)
                print("-" * len(header))
                for fq in all_freqs:
                    row = f"{str(fq) + ' MHz':>8}  "
                    row += "  ".join(
                        f"{series[n][metric].loc[fq]:>{col_w}.3e}" if fq in series[n].index else f"{'N/A':>{col_w}}"
                        for n in series
                    )
                    print(row)


def print_execution_times(experiment_names, benchmarks, freq=None):
    for benchmark in benchmarks:
        series = {}
        for name in experiment_names:
            try:
                series[name] = load_exec_times(f"{BASE}/{name}", benchmark)
            except Exception as exc:
                print(f"[skipped {name}: {exc}]")

        if not series:
            continue

        all_freqs = sorted(set().union(*[set(s.index) for s in series.values()]))
        if freq is not None:
            all_freqs = [freq] if freq in all_freqs else []

        col_w = 14
        header = f"{'freq':>8}  " + "  ".join(f"{n:>{col_w}}" for n in series)
        print(f"\n=== {benchmark} — execution times (s, mean across PEs & runs) ===")
        print(header)
        print("-" * len(header))
        for fq in all_freqs:
            row = f"{str(fq) + ' MHz':>8}  "
            row += "  ".join(
                f"{s.loc[fq]:>{col_w}.2f}" if fq in s.index else f"{'N/A':>{col_w}}"
                for s in series.values()
            )
            print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENTS = [
    os.path.basename(SLACKAWARE_BASE),
    os.path.basename(IMPROVED_SLACKAWARE),
    os.path.basename(IMPROVED_SLACKAWARE_POST_BUGFIX),
]


def _build_parser():
    p = argparse.ArgumentParser(description="DVFS experiment comparison tool")
    p.add_argument(
        "--experiments", "-e",
        nargs="+",
        metavar="FOLDER",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment folder names under BASE (default: all known slack variants)",
    )
    p.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
        metavar="BENCHMARK",
        help=f"Benchmarks to include (default: all — {BENCHMARKS})",
    )
    _display_choices = ["execution-times", "freq-sweep", "comparison", "best-vs-homo"]
    p.add_argument(
        "--measure", "-m",
        choices=_display_choices + list(GOAL_FUNCS.keys()),
        nargs="+",
        default=["comparison", "freq-sweep", "best-vs-homo"],
        help=(
            "What to display (default: comparison freq-sweep best-vs-homo). "
            "Also accepts metric names: " + ", ".join(GOAL_FUNCS.keys())
        ),
    )
    p.add_argument(
        "--freq", "-f",
        type=int,
        default=None,
        metavar="MHZ",
        help="Filter output to a single frequency in MHz (applies to freq-sweep and execution-times)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    measures = set(args.measure)

    if "comparison" in measures:
        for benchmark in args.benchmarks:
            print_comparison(benchmark)

    if "freq-sweep" in measures:
        print()
        for benchmark in args.benchmarks:
            for name in args.experiments:
                df = load_combined_slack_df(f"{BASE}/{name}", benchmark)
                print_freq_sweep(df, name, benchmark, freq=args.freq)

    if "best-vs-homo" in measures:
        print()
        for benchmark in args.benchmarks:
            print_best_vs_homo(benchmark)

    if "execution-times" in measures:
        print()
        print_execution_times(args.experiments, args.benchmarks, freq=args.freq)

    metric_measures = [m for m in args.measure if m in GOAL_FUNCS]
    if metric_measures:
        print()
        print_metric_scores(args.experiments, args.benchmarks, metric_measures, freq=args.freq)
