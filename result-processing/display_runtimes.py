import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/data/users/sargent/dvfs_thesis/local_jobs/L25-cholesky/180371/timings_pe0.csv')
df = df.head(10)

min_ts = df['start_ts'].min()
df['rel_start'] =( df['start_ts'].astype(int) - min_ts) / 1000_000
df['rel_end'] = ( df['end_ts'].astype(int) - min_ts ) / 1000_000 

plt.figure(figsize=(10, 4))
for i, row in df.iterrows():
    plt.hlines(y=row['op_name'], xmin=row['rel_start'], xmax=row['rel_end'], linewidth=8, color='teal')

plt.xlabel('Relative Time (ms)')
plt.ylabel('Operation')
plt.title('Task Execution Intervals')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('timeline.png')