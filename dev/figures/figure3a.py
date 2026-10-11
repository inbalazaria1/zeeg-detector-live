"""
Figure 3a - Agreement with the offline pipeline, with and without the
lookahead correction.

Left: precision, recall and F1 averaged across the three recordings.
Right: F1 per recording. The manuscript reports precision and recall only as
means, so per-recording bars are F1 only - nothing here is interpolated.

Leave-one-recording-out, exact-epoch matching, threshold 0.70, scored from
second 1200 (Table 2 and section 3.2).
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
CORRECTED = "#1E40AF"

# ------------------------------------------------------------------ data
# mean across recordings              frozen   corrected
MEANS = [("precision",                 0.316,   0.388),
         ("recall",                    0.366,   0.392),
         ("F1",                        0.339,   0.383)]
# F1 per recording
PERREC = [("DH3",                      0.334,   0.426),
          ("AO0",                      0.249,   0.268),
          ("MF7",                      0.434,   0.455)]

XS = [0.0, 1.0, 2.0, 3.55, 4.55, 5.55]
W, OFF = 0.34, 0.185

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 68 * MM))
ax = fig.add_axes([0.075, 0.145, 0.895, 0.660])

for x, (_lab, a, b) in zip(XS, MEANS + PERREC):
    ax.bar(x - OFF, a, W, color=FROZEN, edgecolor=FROZEN_E, lw=0.5, zorder=3)
    ax.bar(x + OFF, b, W, color=CORRECTED, edgecolor="none", zorder=3)
    ax.text(x - OFF, a + 0.010, f"{a:.3f}", ha="center", va="bottom",
            fontsize=6.9, color=INK)
    ax.text(x + OFF, b + 0.010, f"{b:.3f}", ha="center", va="bottom",
            fontsize=6.9, color=INK)

# block separator and headers
ax.axvline(2.78, color=GRID, lw=0.8, zorder=1)
ax.text(1.0, 0.548, "mean across recordings", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")
ax.text(4.55, 0.548, "F1 by recording", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")

ax.set_xticks(XS)
ax.set_xticklabels([m[0] for m in MEANS] + [p[0] for p in PERREC],
                   fontsize=8, color=INK)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_ylim(0, 0.60)
ax.set_xlim(-0.62, 6.17)
ax.set_ylabel("agreement with the offline pipeline", fontsize=8, color=MUTED,
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

ax.legend(handles=[Patch(facecolor=FROZEN, edgecolor=FROZEN_E, lw=0.5,
                         label="frozen classifier on the causal chain"),
                   Patch(facecolor=CORRECTED, edgecolor="none",
                         label="+ NREM-trained correction")],
          loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
          frameon=False, fontsize=8, handlelength=1.1, handleheight=0.9,
          columnspacing=1.6, borderpad=0, labelspacing=0.3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure3a")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
