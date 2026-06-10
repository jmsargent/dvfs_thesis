#!/usr/bin/env python3
"""
Analyze retune-vs-transfer samples from retune_latency_hiding (job 159).

Usage:  analyze.py <samples.csv>

CSV: kind,cond,dir,b,mb,sample,ms
  retune   rows: cond in {idle,send,recv}  (GPU state), dir in {up,down}
  transfer rows: cond in {plain,retune}    (concurrent retune?), dir in {send,recv}

Proves/disproves hiding in BOTH directions:
  - retune cost tau per GPU condition  -> does a busy GPU inflate the retune?
  - transfer time plain vs retune      -> does a retune inflate the transfer?
  - tolerance bound on deployment tau, and P(tau > transfer) per b (odds of leak).
Figures (retune_hist.png, transfer_elbow.png) written next to the CSV.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("usage: analyze.py <samples.csv>")
path = sys.argv[1]
outdir = os.path.dirname(os.path.abspath(path))

with open(path) as f:
    lines = f.readlines()
hdr = [i for i, l in enumerate(lines) if l.startswith("kind,cond,dir")]
if not hdr:
    sys.exit(f"no 'kind,cond,dir' header found in {path}")
df = pd.read_csv(io.StringIO("".join(lines[hdr[0]:])), on_bad_lines="skip")
for col in ("b", "mb", "ms"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["ms"])

ret = df[df["kind"] == "retune"]
xfer = df[df["kind"] == "transfer"]
pct = lambda a, q: float(np.percentile(a, q))


def line(a):
    return (f"n={a.size:5d}  mean={a.mean():7.3f}  std={a.std():6.3f}  min={a.min():7.3f}  "
            f"p50={pct(a,50):7.3f}  p99={pct(a,99):7.3f}  p99.9={pct(a,99.9):7.3f}  max={a.max():7.3f}")


# ── Phase 1: retune cost per GPU condition (does a busy GPU inflate tau?) ─────
print("=== retune cost tau (ms) per GPU condition ===")
conds = {}
for cond in ("idle", "send", "recv"):
    a = ret[ret["cond"] == cond]["ms"].to_numpy()
    if a.size:
        conds[cond] = a
        print(f"  {cond:5s}: {line(a)}")

inflight = np.concatenate([conds[c] for c in ("send", "recv") if c in conds]) \
    if ("send" in conds or "recv" in conds) else np.array([])
tau = inflight if inflight.size else conds.get("idle", np.array([]))
if tau.size == 0:
    sys.exit("no retune samples found")

if inflight.size and "idle" in conds:
    idle = conds["idle"]
    dmu = inflight.mean() - idle.mean()
    print(f"  in-flight vs idle: mean {dmu:+.3f} ms, max {inflight.max()-idle.max():+.3f} ms"
          f"  -> {'busy GPU INFLATES tau (idle would underestimate)' if dmu > 0.5 else 'no meaningful inflation'}")
    if inflight.mean() > 3 * idle.mean():
        print("  !! in-flight tau >> idle: setFrequency may be SERIALIZING behind the copy "
              "(retune cannot overlap -> hiding disproved). Check vs cover-b transfer time.")

M = tau.size
tau_max = float(tau.max())
print(f"\n=== deployment tau ({'in-flight' if inflight.size else 'idle'}), M={M} ===  max={tau_max:.3f} ms")
for p in (0.99, 0.999, 0.9999):
    print(f"  tolerance: >= {100*p:7.4f}% of retunes <= {tau_max:.3f} ms at {100*(1-p**M):7.3f}% confidence")

# ── Phase 2: transfer time, plain vs retune (does a retune inflate transfer?) ─
plain = xfer[xfer["cond"] == "plain"]
rtn = xfer[xfer["cond"] == "retune"]
if len(plain) and len(rtn):
    pj = plain.groupby(["b", "dir"])["ms"].median()
    rj = rtn.groupby(["b", "dir"])["ms"].median()
    d = (rj - pj).dropna()
    print(f"\n=== transfer: retune-in-flight minus plain (median over b,dir) ===")
    print(f"  mean {d.mean():+.3f} ms, max {d.max():+.3f} ms"
          f"  -> {'a concurrent retune SLOWS the transfer' if d.mean() > 0.5 else 'retune does not affect the transfer'}")

# ── P(tau > transfer) per b, using the clean (plain) copy time, both dirs ─────
print("\n=== transfer time per b (plain), and P(retune fails to hide) ===")
base = plain if len(plain) else xfer
sorted_tau = np.sort(tau)
rows = []
for b, g in base.groupby("b"):
    t = g["ms"].to_numpy()
    mb = float(g["mb"].iloc[0])
    smin = g[g.dir == "send"]["ms"].min() if (g.dir == "send").any() else np.nan
    rmin = g[g.dir == "recv"]["ms"].min() if (g.dir == "recv").any() else np.nan
    p_leak = float(((M - np.searchsorted(sorted_tau, t, side="right")) / M).mean())
    margin = float(t.min() - tau_max)
    rows.append((int(b), mb, t.size, float(t.min()), pct(t, 50), float(t.max()),
                 float(smin), float(rmin), p_leak, margin))
    print(f"  b={int(b):6d} ({mb:7.1f} MB): n={t.size:4d}  min={t.min():7.3f}  p50={pct(t,50):7.3f}  "
          f"max={t.max():7.3f}  (send_min={smin:6.2f} recv_min={rmin:6.2f})  |  "
          f"P(leak)={p_leak:.2e}  margin={margin:+.2f} ms")
res = pd.DataFrame(rows, columns=["b", "mb", "n", "min", "p50", "max",
                                  "send_min", "recv_min", "p_leak", "margin"]).sort_values("b")

hidden = res[res["margin"] > 0]
if len(hidden):
    r = hidden.iloc[0]
    print(f"\n=> POSITIVE proved from b={int(r.b)} ({r.mb:.0f} MB): all {int(r.n)} transfers "
          f"({r['min']:.1f} ms min) beat all {M} retunes ({tau_max:.1f} ms max); P(leak)={r.p_leak:.2e}.")
else:
    print("\n=> no swept size cleared the worst retune; raise --b-list top end.")

# ── figures ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 4))
for cond, c in [("idle", "tab:green"), ("send", "tab:blue"), ("recv", "tab:orange")]:
    if cond in conds:
        plt.hist(conds[cond], bins=80, alpha=0.55, label=cond, color=c)
plt.axvline(tau_max, color="k", ls="--", lw=1, label=f"deployment max = {tau_max:.1f} ms")
plt.xlabel("retune host blocking time tau (ms)")
plt.ylabel("count")
plt.title("Retune cost by GPU condition")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "retune_hist.png"), dpi=130)
plt.close()

if len(res):
    plt.figure(figsize=(7, 4.5))
    b = res["b"].to_numpy()
    plt.fill_between([b.min(), b.max()], float(tau.min()), tau_max, color="tab:red",
                     alpha=0.15, label=f"retune tau [{tau.min():.1f}, {tau_max:.1f}] ms")
    plt.axhline(tau_max, color="tab:red", ls="--", lw=1)
    yerr = np.vstack([res["p50"] - res["min"], res["max"] - res["p50"]])
    plt.errorbar(b, res["p50"], yerr=yerr, fmt="o-", capsize=3, color="tab:blue",
                 label="transfer time (p50, min-max)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("panel width b (columns)")
    plt.ylabel("time (ms)")
    plt.title("Transfer hides retune where its band clears the retune max")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "transfer_elbow.png"), dpi=130)
    plt.close()

print(f"\nfigures: {outdir}/retune_hist.png, {outdir}/transfer_elbow.png")
