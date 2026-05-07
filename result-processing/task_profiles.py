


#
# Want to create a new csv file that contain
# op , freq_mhz, energy_mj, executiontime_ns

# there is a basepath to benchmark data: /data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters
# in there there are benchmarks done at frequencies with naming convention <benchmark>_<frequency>
# in that folder there is an energy csv file and a profile csv file for each PE

# lets start by taking an example:
# I would like to check for a single PE, how consistent are runtimes for kernels between runs?


import pandas as pd

df = pd.read_csv('/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040/profile/profile_pe0.csv')

# min_ts = df['start_ts'].min()
# df['rel_start'] = (df['start_ts'].astype(int) - min_ts) / 1_000_000
# df['rel_end']   = (df['end_ts'].astype(int)   - min_ts) / 1_000_000
# df['exec_us']   = (df['end_ts'].astype(int) - df['start_ts'].astype(int)) / 1_000

# # For each task, pivot runtimes across runs and compute CV
# task_runs = df.pivot_table(index='op_name', columns='run', values='exec_us', aggfunc='first')
# task_runs['mean_us'] = task_runs.mean(axis=1)
# task_runs['std_us']  = task_runs.std(axis=1)
# task_runs['cv']      = task_runs['std_us'] / task_runs['mean_us']

# print(task_runs[['mean_us', 'std_us', 'cv']].sort_values('cv', ascending=False))


#
# Want to create a new csv file that contain
# op , freq_mhz, energy_mj, executiontime_ns

# there is a basepath to benchmark data: /data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters
# in there there are benchmarks done at frequencies with naming convention <benchmark>_<frequency>
# in that folder there is an energy csv file and a profile csv file for each PE


def create_database():
    import numpy as np

    freq_mhz = 2040  # encoded in benchmark directory name
    pe = 0

    df_energy  = pd.read_csv('/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040/energy/gpu_0.csv')
    df_profile = pd.read_csv('/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040/profile/profile_pe0.csv')

    # energy: Unix seconds (float); profile: Unix nanoseconds (int) — confirmed ratio ~1e9
    energy_ts = df_energy['timestamp'].to_numpy()
    energy_mj = df_energy['total_energy_mj'].to_numpy()

    start_s = df_profile['start_ts'].to_numpy() / 1_000_000_000
    end_s   = df_profile['end_ts'].to_numpy()   / 1_000_000_000

    energy_at_start = np.interp(start_s, energy_ts, energy_mj)
    energy_at_end   = np.interp(end_s,   energy_ts, energy_mj)

    result = pd.DataFrame({
        'op_name':            df_profile['op_name'],
        "pe" :                pe,
        'freq_mhz':           freq_mhz,
        'energy_consumed_mj': energy_at_end - energy_at_start,
        'executiontime_ns':   df_profile['end_ts'] - df_profile['start_ts'],
    })

    print(result)
    return result

create_database()