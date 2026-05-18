import pandas as pd
import numpy as np
import os


def make_energy_func(df_energy):
    """Returns f(t) -> cumulative energy (mJ) at unix timestamp t (seconds).
    Energy over [t_start, t_end] = f(t_end) - f(t_start).
    """
    ts = df_energy["timestamp"].values
    ej = df_energy["total_energy_mj"].values
    return lambda t: np.interp(t, ts, ej)


def make_edp_func(m=1, n=1):
    return lambda energy_mj, delay_s: (energy_mj**m) * (delay_s**n)


def calculate_goal_functions(energy_funcs, df_profile, goal_func):

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
            f(row["last_task_end"]) - f(row["first_task_start"]) for f in energy_funcs
        ),
        axis=1,
    )

    scores = goal_func(energy, delay)

    return pd.Series(
        {
            "mean": scores.mean(),
            "median": scores.median(),
            "cv": scores.std() / scores.mean(),
        }
    )


def load_benchmark_profiles(
    base, pes, energy_name="gpu_{pe}.csv", profile_name="profile_pe{pe}.csv"
):
    energy_funcs = [
        make_energy_func(pd.read_csv(f"{base}/energy/{energy_name.format(pe=pe)}"))
        for pe in pes
    ]
    df_profile = pd.concat(
        [pd.read_csv(f"{base}/profile/{profile_name.format(pe=pe)}") for pe in pes]
    )
    return energy_funcs, df_profile


def get_available_freqs(HOMO_BASE, BENCHMARK):
    freqs = sorted(
        int(d.split("_")[1])
        for d in os.listdir(HOMO_BASE)
        if d.startswith(f"{BENCHMARK}_")
    )

    return freqs


HOMO_BASE = "/Users/jonathansargent/dvfs_thesis/homogenous-retune-best-parameters"
BENCHMARK = "cholesky"

freqs = get_available_freqs(HOMO_BASE, BENCHMARK)

GOAL_FUNCS = {
    "edp": make_edp_func(),
    "ed2p": make_edp_func(n=2),
    "e2dp": make_edp_func(m=2),
    "energy": make_edp_func(n=0),
}

rows = []
for freq in freqs:
    energy_funcs, df_profile = load_benchmark_profiles(
        f"{HOMO_BASE}/{BENCHMARK}_{freq}", [0, 1, 2, 3]
    )
    row = {"freq": freq}
    for name, goal_func in GOAL_FUNCS.items():
        row[name] = calculate_goal_functions(energy_funcs, df_profile, goal_func)[
            "mean"
        ]
    rows.append(row)

df_homo = pd.DataFrame(rows).set_index("freq")

SYNC_BASE = "/Users/jonathansargent/dvfs_thesis/retune_on_sync2"

sync_rows = {}
for name, goal_func in GOAL_FUNCS.items():
    energy_funcs, df_profile = load_benchmark_profiles(
        f"{SYNC_BASE}/{BENCHMARK}/{name}",
        [0, 1, 2, 3],
        profile_name="tasks_pe{pe}.csv",
    )
    sync_rows[name] = calculate_goal_functions(energy_funcs, df_profile, goal_func)[
        "mean"
    ]

for name in GOAL_FUNCS:
    best_freq = df_homo[name].idxmin()
    best_homo = df_homo.loc[best_freq, name]
    sync = sync_rows[name]
    ratio = best_homo / sync 
    print(
        f"{name}: best homo = {best_homo:.3e} @ {best_freq} MHz | "
        f"sync retune = {sync:.3e} | ratio (over > 1 is improvement for sync) = {ratio:.2f}x"
    )
