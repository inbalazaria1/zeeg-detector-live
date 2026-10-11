"""
Figure 3b - Agreement as a function of how tightly a detection has to line up
with the offline one.

Left: precision. Right: F1. Both are means across the three recordings, frozen
classifier versus the NREM-trained correction, at four match windows - from
exact-epoch agreement out to +/-2 epochs (+/-500 ms).

The correction is ahead of the frozen chain at every window, and the gap it
opens at the exact epoch survives loosening the criterion. Nothing here is
interpolated: the exact-epoch column is the same pair of numbers as figure 3a.

Leave-one-recording-out, threshold 0.70, scored from second 1200.
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
# match window                 precision            F1
#                              frozen  corrected    frozen  corrected
ROWS = [
    ("same\nepoch",             0.316,  0.388,      0.339,  0.383),
    ("this epoch\nor the next", 0.369,  0.451,      0.396,  0.444),
    ("±1\nepoch",          0.445,  0.513,      0.477,  0.507),
    ("±2\nepochs",         0.516,  0.594,      0.553,  0.587),
]

XS_P = [0.0, 1.0, 2.0, 3.0]                  # precision block
XS_F = [4.55, 5.55, 6.55, 7.55]              # F1 block
W, OFF = 0.34, 0.185

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 72 * MM))
ax = fig.add_axes([0.070, 0.185, 0.900, 0.625])

for xs, (a_i, b_i) in ((XS_P, (1, 2)), (XS_F, (3, 4))):
    for x, row in zip(xs, ROWS):
        a, b = row[a_i], row[b_i]
        ax.bar(x - OFF, a, W, color=FROZEN, edgecolor=FROZEN_E, lw=0.5, zorder=3)
        ax.bar(x + OFF, b, W, color=CORRECTED, edgecolor="none", zorder=3)
        ax.text(x - OFF, a + 0.010, f"{a:.3f}", ha="center", va="bottom",
                fontsize=6.9, color=INK)
        ax.text(x + OFF, b + 0.010, f"{b:.3f}", ha="center", va="bottom",
                fontsize=6.9, color=INK)

# block separator and headers
ax.axvline(3.78, color=GRID, lw=0.8, zorder=1)
ax.text(1.5, 0.648, "precision", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")
ax.text(6.05, 0.648, "F1", ha="center", va="bottom",
        fontsize=8, color=INK, fontweight="bold")

ax.set_xticks(XS_P + XS_F)
ax.set_xticklabels([r[0] for r in ROWS] * 2, fontsize=7.4, color=INK,
                   linespacing=1.15)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim(0, 0.70)
ax.set_xlim(-0.62, 8.17)
ax.set_ylabel("agreement with the offline pipeline", fontsize=8, color=MUTED,
              labelpad=5)
ax.tick_params(axis="both", length=2.5, width=0.6, color=MUTED,
               labelsize=7.4, labelcolor=MUTED, pad=2)
ax.tick_params(axis="x", length=0, labelsize=7.4, labelcolor=INK, pad=3)
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
    ax.spines[s].set_linewidth(0.6)

TR = ax.get_xaxis_transform()
ax.text(3.78, -0.215, "match window  (250 ms per epoch)", transform=TR,
        ha="center", va="top", fontsize=7.2, color=MUTED, style="italic",
        clip_on=False)

ax.legend(handles=[Patch(facecolor=FROZEN, edgecolor=FROZEN_E, lw=0.5,
                         label="frozen classifier on the causal chain"),
                   Patch(facecolor=CORRECTED, edgecolor="none",
                         label="+ NREM-trained correction")],
          loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
          frameon=False, fontsize=8, handlelength=1.1, handleheight=0.9,
          columnspacing=1.6, borderpad=0, labelspacing=0.3)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure3b")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
