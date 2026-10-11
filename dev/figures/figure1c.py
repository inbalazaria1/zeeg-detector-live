"""
Figure 1c - Timing of a single decision.

One 250 ms epoch period. After the epoch closes the system waits 25 ms of filter
lookahead, computes for 10.9 ms, and decides at 35.9 ms - inside the 50 ms
latency budget and a small fraction of the epoch.

Component-level compute times are in Table 1 and deliberately not repeated here.
Medians over 22,671 timed windows.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK, MUTED = "#1F2328", "#6B7280"
LOOK, COMPUTE, LIMIT = "#C7DAF7", "#2563EB", "#C2410C"

LOOKAHEAD, COMPUTE_T = 25.0, 10.89
DECISION = LOOKAHEAD + COMPUTE_T                 # 35.89, reported as 35.9
BUDGET, EPOCH = 50.0, 250.0

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 34 * MM))
ax = fig.add_axes([0.055, 0.40, 0.915, 0.38])

YLO, YHI = -1.5, 2.1
BH = (YHI - YLO) * 0.28

ax.add_patch(Rectangle((0, -BH / 2), LOOKAHEAD, BH, facecolor=LOOK,
                       edgecolor="none", zorder=3))
ax.add_patch(Rectangle((LOOKAHEAD, -BH / 2), COMPUTE_T, BH, facecolor=COMPUTE,
                       edgecolor="none", zorder=3))

ax.axvline(BUDGET, color=LIMIT, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.axvline(EPOCH, color=MUTED, lw=0.9, zorder=4)
ax.text(BUDGET + 4, 0.95, "50 ms latency budget", ha="left", va="bottom",
        fontsize=8, color=LIMIT)
ax.text(EPOCH - 4, 0.95, "250 ms — next epoch closes", ha="right", va="bottom",
        fontsize=8, color=MUTED)

# the decomposition reads inline, level with the bar it describes
ax.text(58, 0, "decision at 35.9 ms   =   25 ms lookahead  +  10.9 ms compute",
        ha="left", va="center", fontsize=8, color=INK)

ax.set_xlim(-8, 262)
ax.set_ylim(YLO, YHI)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.spines["bottom"].set_linewidth(0.6)
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.tick_params(axis="x", length=2.5, width=0.6, color=MUTED,
               labelsize=7.4, labelcolor=MUTED, pad=2)
ax.set_xlabel("time after the epoch closes (ms)", fontsize=8, color=MUTED,
              labelpad=3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure1c")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
