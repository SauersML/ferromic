#!/usr/bin/env python
"""Figure 2: extended-flux breakdown curves (rho=1e-8)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# main sweep (flux 0..1e-6) + extreme sweep (1e-6..1e-4), rho=1e-8
main = json.load(open("sweep_full.json"))
ext = json.load(open("sweep_extreme.json"))

def series(scenario, depth, metric):
    pts = {}
    for r in main + ext:
        if r["scenario"] == scenario and r["depth"] == depth and abs(r["rho"]-1e-8) < 1e-30:
            pts[r["m_flux"]] = r[metric]
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xfloor = 3e-10
colors = {"young": "tab:green", "recent": "tab:blue"}

# panel A: call rate (FPR for single, power for recurrent)
ax = axes[0]
for depth in ("young", "recent"):
    xs, ys = series("single", depth, "recurrent_call_rate")
    ax.plot([max(x, xfloor) for x in xs], ys, "o-", color=colors[depth],
            label=f"single (FPR), {depth}")
    xs, ys = series("recurrent", depth, "recurrent_call_rate")
    ax.plot([max(x, xfloor) for x in xs], ys, "s--", color=colors[depth],
            label=f"recurrent (power), {depth}")
ax.set_xscale("log"); ax.set_xlabel("between-orientation flux m (per lineage per gen)")
ax.set_ylabel("recurrent-call rate"); ax.set_ylim(-0.03, 1.03)
ax.axhline(0.05, color="gray", lw=0.7, ls=":")
ax.axvspan(0, 1e-6, color="green", alpha=0.06)
ax.set_title("A. Classifier call rate vs flux (rho=1e-8)\nshaded = plausible GC range")
ax.legend(fontsize=8)

# panel B: mean inferred number of events
ax = axes[1]
for depth in ("young", "recent"):
    xs, ys = series("single", depth, "mean_events")
    ax.plot([max(x, xfloor) for x in xs], ys, "o-", color=colors[depth],
            label=f"single (truth=1), {depth}")
    xs, ys = series("recurrent", depth, "mean_events")
    ax.plot([max(x, xfloor) for x in xs], ys, "s--", color=colors[depth],
            label=f"recurrent (truth=3), {depth}")
ax.set_xscale("log"); ax.set_xlabel("between-orientation flux m (per lineage per gen)")
ax.set_ylabel("mean inferred # inversion events")
ax.axhline(1, color="gray", lw=0.6, ls=":"); ax.axhline(3, color="gray", lw=0.6, ls=":")
ax.axvspan(0, 1e-6, color="green", alpha=0.06)
ax.set_title("B. Inferred event count inflates under extreme flux")
ax.legend(fontsize=8)

fig.suptitle("Between-orientation flux (gene conversion / double crossover): "
             "classifier robust up to m~1e-6, breaks down beyond", fontsize=12)
fig.tight_layout()
fig.savefig("flux_breakdown.png", dpi=150)
print("wrote flux_breakdown.png")
