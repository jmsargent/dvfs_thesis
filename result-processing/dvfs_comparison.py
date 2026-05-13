import pandas as pd
import numpy as np



def make_energy_func(df_energy):
    """Returns f(t) -> cumulative energy (mJ) at unix timestamp t (seconds).
    Energy over [t_start, t_end] = f(t_end) - f(t_start).
    """
    ts = df_energy['timestamp'].values
    ej = df_energy['total_energy_mj'].values
    return lambda t: np.interp(t, ts, ej)

def make_edp_func(m=1, n=1):
    return lambda energy_mj, delay_s: (energy_mj ** m) * (delay_s ** n)


def calculate_goal_functions(energy_funcs, df_profile, goal_func):

    run_bounds = df_profile.groupby('run').agg(
        first_task_start=('start_ts', 'min'),
        last_task_end=('end_ts', 'max'),
    ) / 1e9

    delay = run_bounds['last_task_end'] - run_bounds['first_task_start']
    energy = run_bounds.apply(
        lambda row: sum(
            f(row['last_task_end']) - f(row['first_task_start']) for f in energy_funcs
        ),
        axis=1,
    )

    scores = goal_func(energy, delay)

    return pd.Series({'mean': scores.mean(), 'median': scores.median(), 'cv': scores.std() / scores.mean()})


def load_benchmark_profiles(base, pes, energy_name='gpu_{pe}.csv', profile_name='profile_pe{pe}.csv'):
    energy_funcs = [
        make_energy_func(pd.read_csv(f'{base}/energy/{energy_name.format(pe=pe)}'))
        for pe in pes
    ]
    df_profile = pd.concat([
        pd.read_csv(f'{base}/profile/{profile_name.format(pe=pe)}')
        for pe in pes
    ])
    return energy_funcs, df_profile


# HOMOGENOUS_BASE = '/Users/jonathansargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040'
# SYNC_RETUNE_SCORE = '/Users/jonathansargent/dvfs_thesis/retune_on_sync2/cholesky/edp'

# goal_func = make_edp_func(m=1, n=1)

# energy_funcs_homo, df_profile_homo = load_benchmark_profiles(HOMOGENOUS_BASE, [0, 1, 2, 3])
# homogenous_score = calculate_goal_functions(energy_funcs_homo, df_profile_homo, goal_func)

# energy_funcs_sync, df_profile_sync = load_benchmark_profiles(SYNC_RETUNE_SCORE, [0, 1, 2, 3], profile_name='tasks_pe{pe}.csv')
# sync_score = calculate_goal_functions(energy_funcs_sync, df_profile_sync, goal_func)

# print("homogenous")
# print(homogenous_score)
# print("sync")
# print(sync_score)

