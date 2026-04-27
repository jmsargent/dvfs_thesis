import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# csv_file = "lu_mustard.csv"
csv_file = "cholesky_mustard.csv"

freq, median_time, energy, edp = [], [], [], []
with open(csv_file, newline="") as f:
    for row in csv.DictReader(f):
        freq.append(int(row["freq_mhz"]))
        median_time.append(float(row["median_time_s"]))
        energy.append(float(row["total_energy_mj"]))
        edp.append(float(row["edp"]))

fig, axes = plt.subplots(3, 1, figsize=(9, 12))
fig.suptitle("cholesky — Benchmark Results", fontsize=14, fontweight="bold", y=0.98)

configs = [
    (axes[0], median_time, "Median Time (s)",   "#2196F3", "Median Execution Time vs Frequency"),
    (axes[1], energy,      "Total Energy (mJ)", "#E91E63", "Total Energy vs Frequency"),
    (axes[2], edp,         "EDP (mJ·s)",        "#FF9800", "Energy–Delay Product vs Frequency"),
]

for ax, y, ylabel, color, title in configs:
    ax.plot(freq, y, color=color, linewidth=1.8, marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.5)

    # Mark minimum
    min_idx = y.index(min(y))
    min_x, min_y = freq[min_idx], y[min_idx]
    ax.plot(min_x, min_y, marker="*", markersize=13, color="red", zorder=5)
    ax.annotate(f"min @ {min_x} MHz\n{min_y:,.4g}", xy=(min_x, min_y),
                xytext=(12, 12), textcoords="offset points",
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("Frequency (MHz)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(min(freq) - 50, max(freq) + 50)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("benchmark_plots.png", dpi=150, bbox_inches="tight")
