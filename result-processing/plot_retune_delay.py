#!/usr/bin/env python3
"""Plot retune latency models for L4 and L40S GPUs."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def l4_ramp_down_ns(delta):
    if delta <= 555:
        return 930 * delta + 168020
    elif delta <= 1208:
        return 682910
    else:
        return 3290 * delta - 3296400


def l4_ramp_up_ns(delta):
    return 315000


def l40s_ramp_down_ns(delta):
    if delta < 550:
        return 3060 * delta + 89140
    elif delta < 2076:
        return 1783190
    else:
        return 9590 * (delta - 2076) + 1783190


def l40s_ramp_up_ns(delta):
    if delta < 570:
        return 600 * delta + 267940
    elif delta < 1055:
        return 610580
    else:
        return 610580 + 970 * (delta - 1055)


def to_us(ns):
    return ns / 1000.0


MAX_DELTA_L4   = 1500
MAX_DELTA_L40S = 2500

deltas_l4   = np.arange(1, MAX_DELTA_L4 + 1)
deltas_l40s = np.arange(1, MAX_DELTA_L40S + 1)

l4_down   = np.array([to_us(l4_ramp_down_ns(d))   for d in deltas_l4])
l4_up     = np.array([to_us(l4_ramp_up_ns(d))     for d in deltas_l4])
l40s_down = np.array([to_us(l40s_ramp_down_ns(d)) for d in deltas_l40s])
l40s_up   = np.array([to_us(l40s_ramp_up_ns(d))   for d in deltas_l40s])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("GPU Frequency Retune Latency Model", fontsize=14)

for ax, gpu, deltas, down, up in [
    (axes[0], "L4",   deltas_l4,   l4_down,   l4_up),
    (axes[1], "L40S", deltas_l40s, l40s_down, l40s_up),
]:
    ax.plot(deltas, down, label="Ramp down (higher→lower)", color="tab:blue")
    ax.plot(deltas, up,   label="Ramp up (lower→higher)",   color="tab:orange")
    ax.set_title(f"NVIDIA {gpu}")
    ax.set_xlabel("Frequency delta (MHz)")
    ax.set_ylabel("Retune latency (µs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.show()
