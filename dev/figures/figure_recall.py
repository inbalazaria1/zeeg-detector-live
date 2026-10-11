"""
Figure - Precision at matched recall, for the frozen classifier and two
correction training sets.

The point a single operating point cannot make: the correction is better at
every recall it is compared at, and the frozen classifier cannot reach recall
0.45 on DH3 at any threshold whereas the NREM-trained correction reaches it at
precision 0.424. Rescaling scores cannot extend a curve to a recall it never
attained, so the gain is in the ordering of epochs, not in where the boundary
sits.

Left block: mean over the three recordings. Right block: per recording at
recall 0.45. Colours match Figure 3 - light blue is the frozen classifier, navy
the shipped correction.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK, MUTED, GRID = "#1F2328", "#6B7280", "#E3E6EA"
FROZEN, FROZEN_E = "#60A5FA", "#2563EB"      # light bar carries an edge for print
CROPS = "#2563EB"
NREM = "#1E40AF"                             # the shipped correction, as in Figure 3

# ------------------------------------------------------------------ data
# None marks a recall the detector never attains, at any threshold.
SERIES = [
    ("frozen classifier",                  FROZEN, FROZEN_E,
     [0.362, 0.333, None,  0.455]),
    ("correction on the evaluation crops", CROPS,  "none",
     [0.415, 0.384, None,  0.514]),
    ("correction on all donor NREM",       NREM,   "none",
     [0.424, 0.402, 0.424, 0.556]),
]
GROUPS = ["recall ≥ 0.25", "recall ≥ 0.30", "DH3", "MF7"]

XS = [0.0, 1.0, 2.55, 3.55]                  # two blocks, split after the means
W, OFF = 0.255, 0.27

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 74 * MM))
ax = fig.add_axes([0.075, 0.130, 0.895, 0.700])

for k, (_name, col, ec, vals) in enumerate(SERIES):
    dx = (k - 1) * OFF
    for x, v in zip(XS, vals):
        if v is None:                        # the slot stays empty and says why
            ax.text(x + dx, 0.022, "unreachable at any threshold", rotation=90,
                    ha="center", va="bottom", fontsize=6.8, color=MUTED,
                    zorder=3)
            continue
        ax.bar(x + dx, v, W, color=col, edgecolor=ec,
               lw=0.5 if ec != "none" else 0, zorder=3)
        ax.text(x + dx, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.9, color=INK, zorder=4)

# block separator and headers
ax.axvline(1.78, color=GRID, lw=0.8, zorder=1)
ax.text(0.5, 0.632, "mean across recordings", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")
ax.text(3.05, 0.632, "per recording, at recall 0.45", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")

# the empty slots are the result, so the label sits in them rather than in a note
ax.set_xticks(XS)
ax.set_xticklabels(GROUPS, fontsize=8, color=INK)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim(0, 0.70)
ax.set_xlim(-0.62, 4.17)
ax.set_ylabel("precision at or above that recall", fontsize=8, color=MUTED,
              labelpad=5)
ax.tick_params(axis="both", length=2.5, width=0.6, color=MUTED,
               labelsize=7.4, labelcolor=MUTED, pad=2)
ax.tick_params(axis="x", length=0, labelsize=8, labelcolor=INK)
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
    ax.spines[s].set_linewidth(0.6)

ax.legend(handles=[Patch(facecolor=c, edgecolor=e, lw=0.5 if e != "none" else 0,
                         label=n) for n, c, e, _ in SERIES],
          loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False,
          fontsize=8, handlelength=1.1, handleheight=0.9, columnspacing=1.6,
          borderpad=0, labelspacing=0.3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_recall")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
