import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/Users/jonathansargent/dvfs_thesis/experiments/profiling-nvshmem")

WAIT_KERNEL   = "%kernel_wait_static%"
SIGNAL_KERNEL = "%kernel_signal_static%"

def query_kernels(db: Path, pattern: str) -> pd.Series:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT (k.end - k.start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE s.value LIKE ?
    """, con, params=(pattern,))
    con.close()
    return df["duration_ns"]

def query_p2p(db: Path) -> pd.Series:
    con = sqlite3.connect(db)
    df = pd.read_sql_query("""
        SELECT (end - start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 10
    """, con)
    con.close()
    return df["duration_ns"]

def load_from_sqlite(kernel_pattern: str):
    rows = []
    for db in sorted(BASE.glob("nsys_*/*/*.sqlite")):
        bench = db.parts[-3].replace("nsys_", "")
        freq  = int(db.parts[-2])
        rank  = int(db.stem.replace("rank_", ""))
        dur   = query_kernels(db, kernel_pattern)
        if len(dur):
            rows.append(pd.DataFrame({
                "duration_ns": dur,
                "bench": bench,
                "freq": freq,
                "rank": rank,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_p2p():
    rows = []
    for db in sorted(BASE.glob("nsys_*/*/*.sqlite")):
        bench = db.parts[-3].replace("nsys_", "")
        freq  = int(db.parts[-2])
        rank  = int(db.stem.replace("rank_", ""))
        dur   = query_p2p(db)
        if len(dur):
            rows.append(pd.DataFrame({
                "duration_ns": dur,
                "bench": bench,
                "freq": freq,
                "rank": rank,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── Signal / wait latency (kernel_wait_static) ────────────────────────────────
waits = load_from_sqlite(WAIT_KERNEL)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, bench in zip(axes, ["lu", "cholesky"]):
    sub = waits[waits["bench"] == bench]
    print(f"\n[wait latency] bench={bench}  total={len(sub)}")
    for freq, grp in sub.groupby("freq"):
        print(f"  freq={freq} MHz  n={len(grp)}")
        ax.hist(grp["duration_ns"] / 1_000, bins=80, alpha=0.6,
                density=True, label=f"{freq} MHz")

    baseline = BASE / "signal_latency" / "signal_latency.csv"
    if baseline.exists():
        ax.hist(pd.read_csv(baseline)["latency_ns"] / 1_000, bins=80,
                alpha=0.6, density=True, color="black", label="baseline (idle)")

    ax.set_xlabel("Duration (µs)")
    ax.set_ylabel("Density")
    ax.set_title(f"{bench} — kernel_wait_static")
    ax.legend()

plt.tight_layout()
plt.show()

# ── Tile pull latency (P2P memcpy) ────────────────────────────────────────────
p2p = load_p2p()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, bench in zip(axes, ["lu", "cholesky"]):
    sub = p2p[p2p["bench"] == bench]
    print(f"\n[tile pull / P2P] bench={bench}  total={len(sub)}")
    for freq, grp in sub.groupby("freq"):
        print(f"  freq={freq} MHz  n={len(grp)}")
        ax.hist(grp["duration_ns"] / 1_000, bins=40, alpha=0.6,
                density=True, label=f"{freq} MHz")

    baseline = BASE / f"tile_pull_{bench}" / f"tile_pull_{bench}.csv"
    if baseline.exists():
        ax.hist(pd.read_csv(baseline)["latency_ms"] * 1_000, bins=40,
                alpha=0.6, density=True, color="black", label="baseline (idle)")

    ax.set_xlabel("Duration (µs)")
    ax.set_ylabel("Density")
    ax.set_title(f"{bench} — tile pull (P2P memcpy)")
    ax.legend()

plt.tight_layout()
plt.show()
