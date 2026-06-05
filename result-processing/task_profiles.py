import os
import pandas as pd
import numpy as np

# df = pd.read_csv('/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040/profile/profile_pe0.csv')

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


def create_database(base, benchmark, pe, freq_mhz):

    df_energy  = pd.read_csv(f'{base}/{benchmark}/constant/{freq_mhz}/energy/gpu_{pe}.csv')
    df_profile = pd.read_csv(f'{base}/{benchmark}/constant/{freq_mhz}/profile/tasks_pe{pe}.csv')
    
    
    ##################################################                                               
    # EXTRACT INTERPOLATED ENERGIES BETWEEN TIMESTAMPS 
    ##################################################                                                 
    
    # energy: Unix seconds (float); profile: Unix nanoseconds (int) — confirmed ratio ~1e9
    energy_ts = df_energy['timestamp'].to_numpy()
    energy_uj = df_energy['total_energy_mj'].to_numpy() * 1_000

    # The hardware counter updates less frequently than the logger samples,
    # leaving runs of identical readings. Deduplicate so each interpolation
    # segment spans a real sensor step and ops get a prorated share of it.
    step_mask = np.concatenate([[True], np.diff(energy_uj) != 0])
    energy_ts = energy_ts[step_mask]
    energy_uj = energy_uj[step_mask]

    start_s = df_profile['start_ts'].to_numpy() / 1_000_000_000
    end_s   = df_profile['end_ts'].to_numpy()   / 1_000_000_000

    energy_at_start = np.interp(start_s, energy_ts, energy_uj)
    energy_at_end   = np.interp(end_s,   energy_ts, energy_uj)
    
    ##################################################                                               
    # Obtain waittimes - average between runs
    ##################################################
    
    wait_ns = (df_profile['start_ts'] - df_profile['wait_start_ts']).where(
        df_profile['wait_start_ts'] != 0, other=0
    )
    avg_waittimes = wait_ns.groupby(df_profile['op_name']).transform('mean').round().astype(int)

    result = pd.DataFrame({
        'op_name':            df_profile['op_name'],
        "pe" :                pe,
        'freq_mhz':           freq_mhz,
        'energy_consumed_uj': (energy_at_end - energy_at_start).round().astype(int),
        'executiontime_ns':   df_profile['end_ts'] - df_profile['start_ts'],
        'average_waittime_ns': avg_waittimes,
    })

    return result



def create_dbs(base):

    benchs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )

    PES = [0, 1, 2, 3]

    for bench in benchs:
        constant_dir = os.path.join(base, bench, 'constant')
        freqs = sorted(
            int(d) for d in os.listdir(constant_dir)
            if os.path.isdir(os.path.join(constant_dir, d)) and d.isdigit()
        )

        dfs = []
        for pe in PES:
            for freq in freqs:
                dfs.append(create_database(base, bench, pe, freq))

        pd.concat(dfs, ignore_index=True).to_csv(f'{bench}_database.csv', index=False)

BASE_DIR = '/Users/jonathansargent/dvfs_thesis/experiments/saturate-functional-units'
create_dbs(BASE_DIR)