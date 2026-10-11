"""
Figure - Mean F1 for complete deployable decision rules (Table 4).

Each row is one rule applied to one training set, plotted as the two independent
seed bags joined by a line. The vertical reference is the frozen classifier with
no correction, for which the threshold rule has no effect.

The pattern: holding the threshold at 0.70 lands rightmost in both training sets
where the correction saw NREM periods, and loses in both bags of the smallest
training set.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK, MUTED, GRID = "#1F2328", "#6B7280", "#E3E6EA"
HOLD, OTHER = "#2563EB", "#8A9099"
BASELINE = 0.339

# ------------------------------------------------------------------ data
RULES = ["hold at 0.70", "optimize on donors",
         "anchor on own block (count)", "anchor on own block (fraction)"]

GROUPS = [
    ("evaluation crops", [
        (0.346, 0.343), (0.347, 0.337), (0.353, 0.365), (0.343, 0.365)]),
    ("all donor NREM", [
        (0.368, 0.369), (0.333, 0.366), (0.322, 0.313), (0.333, 0.301)]),
    ("all donor NREM, seed dropped", [
        (0.378, 0.379), (0.364, 0.365), (0.366, 0.341), (0.372, 0.368)]),
]

# ------------------------------------------------------------------ layout
HEAD, ROW, GAP = 1.05, 0.88, 0.55
y, rows, heads = 0.0, [], []
for gname, vals in GROUPS:
    y -= HEAD
    heads.append((y, gname))
    for rule, (b1, b2) in zip(RULES, vals):
        y -= ROW
        rows.append((y, rule, b1, b2))
    y -= GAP
YMIN = y

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 104 * MM))
ax = fig.add_axes([0.315, 0.095, 0.635, 0.795])
LAB = ax.get_yaxis_transform()

ax.axvline(BASELINE, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)

for (yy, rule, b1, b2) in rows:
    c = HOLD if rule == "hold at 0.70" else OTHER
    ax.plot([b1, b2], [yy, yy], color=c, lw=1.0, alpha=0.55,
            solid_capstyle="round", zorder=3)
    ax.plot([b1, b2], [yy, yy], marker="o", ms=4.6, color=c, ls="none",
            mec="white", mew=0.6, zorder=4)
    ax.text(-0.020, yy, rule, transform=LAB, ha="right", va="center",
            fontsize=7.8, color=INK if c == HOLD else MUTED, clip_on=False)
    if rule == "hold at 0.70":                       # selective direct labels
        ax.text(max(b1, b2) + 0.0035, yy, f"{b1:.3f} / {b2:.3f}",
                ha="left", va="center", fontsize=7.2, color=HOLD)

for yy, gname in heads:
    ax.text(-0.020, yy, gname, transform=LAB, ha="right", va="center",
            fontsize=8.2, color=INK, fontweight="bold", clip_on=False)

ax.text(BASELINE - 0.0025, 0.30,
        "frozen classifier, no correction   0.339", ha="right", va="bottom",
        fontsize=7.4, color=MUTED)

ax.set_xlim(0.288, 0.400)
ax.set_ylim(YMIN - 0.15, 0.95)
ax.set_xticks([0.30, 0.32, 0.34, 0.36, 0.38, 0.40])
ax.set_yticks([])
ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.spines["bottom"].set_linewidth(0.6)
ax.tick_params(axis="x", length=2.5, width=0.6, color=MUTED,
               labelsize=7.4, labelcolor=MUTED, pad=2)
ax.set_xlabel("mean F1   (each row shows two independent seed bags)",
              fontsize=8, color=MUTED, labelpad=4)

ax.legend(handles=[
    Line2D([], [], color=HOLD, marker="o", ms=4.6, lw=1.0, mec="white",
           mew=0.6, label="threshold held at 0.70"),
    Line2D([], [], color=OTHER, marker="o", ms=4.6, lw=1.0, mec="white",
           mew=0.6, label="threshold tuned or anchored")],
    loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False,
    fontsize=8, handlelength=1.6, columnspacing=2.0, borderpad=0)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_rules")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
