import pandas as pd
import matplotlib.pyplot as plt

DATABASES = {
    'cholesky': 'cholesky_database.csv',
    'lu':       'lu_database.csv',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, path) in zip(axes, DATABASES.items()):
    df = pd.read_csv(path)
    ax.hist(df['energy_consumed_uj'], bins=100)
    ax.set_title(name)
    ax.set_xlabel('Energy consumed (µJ)')
    ax.set_ylabel('Number of operations')

fig.tight_layout()
plt.savefig('energy_histograms.png', dpi=150)
plt.show()
