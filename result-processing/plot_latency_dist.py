import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

# Load all per-rank trace CSVs, parse bench/freq from path
def load_traces():
    rows = []
    for csv in sorted(BASE.glob("nsys_*/*/*.csv")):
        if "trace_cuda" in csv.name:  # skip old duplicate
            continue
        bench = csv.parts[-3].replace("nsys_", "")
        freq  = int(csv.parts[-2])
        df = pd.read_csv(csv)
        df["bench"] = bench
        df["freq"]  = freq
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

traces = load_traces()

# ── Signal / wait latency ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, bench in zip(axes, ["lu", "cholesky"]):
    sub = traces[
        (traces["bench"] == bench) &
        traces["Name"].str.contains("barrier_on_stream", na=False)
    ]
    for freq, grp in sub.groupby("freq"):
        ax.hist(grp["Duration (ns)"] / 1_000, bins=80, alpha=0.6,
                density=True, label=f"{freq} MHz")

    baseline = BASE / "signal_latency" / "signal_latency.csv"
    if baseline.exists():
        ax.hist(pd.read_csv(baseline)["latency_ns"] / 1_000, bins=80,
                alpha=0.6, density=True, color="black", label="baseline (idle)")

    ax.set_xlabel("Duration (µs)")
    ax.set_ylabel("Density")
    ax.set_title(f"{bench} — nvshmem_int_wait_until")
    ax.legend()

plt.tight_layout()
plt.savefig(BASE / "signal_latency_dist.png", dpi=150)
print("saved signal_latency_dist.png")

# ── Tile pull latency ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, bench in zip(axes, ["lu", "cholesky"]):
    sub = traces[
        (traces["bench"] == bench) &
        (traces["Name"] == "[CUDA memcpy Peer-to-Peer]")
    ]
    for freq, grp in sub.groupby("freq"):
        ax.hist(grp["Duration (ns)"] / 1_000, bins=40, alpha=0.6,
                density=True, label=f"{freq} MHz")

    baseline = BASE / f"tile_pull_{bench}" / f"tile_pull_{bench}.csv"
    if baseline.exists():
        ax.hist(pd.read_csv(baseline)["latency_ms"] * 1_000, bins=40,
                alpha=0.6, density=True, color="black", label="baseline (idle)")

    ax.set_xlabel("Duration (µs)")
    ax.set_ylabel("Density")
    ax.set_title(f"{bench} — tile pull (cudaMemcpy2DAsync P2P)")
    ax.legend()

plt.tight_layout()
plt.savefig(BASE / "tile_pull_dist.png", dpi=150)
print("saved tile_pull_dist.png")
