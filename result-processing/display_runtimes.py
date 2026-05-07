import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# df = pd.read_csv('/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters/cholesky_2040/profile/profile_pe0.csv')
# df = df.head(10)

# min_ts = df['start_ts'].min()
# df['rel_start'] =( df['start_ts'].astype(int) - min_ts) / 1000_000
# df['rel_end'] = ( df['end_ts'].astype(int) - min_ts ) / 1000_000 

# plt.figure(figsize=(10, 4))
# for i, row in df.iterrows():
#     plt.hlines(y=row['op_name'], xmin=row['rel_start'], xmax=row['rel_end'], linewidth=8, color='teal')

# plt.xlabel('Relative Time (ms)')
# plt.ylabel('Operation')
# plt.title('Task Execution Intervals')
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig('timeline.png')


#
# 
#
def load_runtimes(file):
    df = pd.read_csv(file)
    return (df['end_ts'].astype(int) - df['start_ts'].astype(int)) / 1_000_000


def runtime_histogram(runtimes, out, bins, ylim):
    plt.figure(figsize=(10, 5))
    plt.hist(runtimes, bins=bins)
    plt.xlim(bins[0], bins[-1])
    plt.ylim(ylim)
    plt.xlabel('executiontime (ms)')
    plt.ylabel('Count')
    plt.title('Task executiontime')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


BASE = '/data/users/sargent/dvfs_thesis/homogenous-retune-best-parameters'
for benchmark in ['cholesky', 'lu']:
    data = {freq: load_runtimes(f'{BASE}/{benchmark}_{freq}/profile/profile_pe0.csv')
            for freq in [240, 2040]}
    all_runtimes = pd.concat(data.values())
    xlim = (all_runtimes.min(), all_runtimes.max())
    bins = np.linspace(*xlim, 31)
    ylim = (0, max(np.histogram(rt, bins=bins)[0].max() for rt in data.values()) * 1.05)
    for freq, runtimes in data.items():
        runtime_histogram(runtimes, f'histogram_executiontimes_{benchmark}_{freq}.png', bins, ylim)
