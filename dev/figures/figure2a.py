"""
Figure 2a - Median latency per 250 ms decision, decomposed by component.

Top: the decision timeline on a true linear axis. Declared filter lookahead
(25 ms) is charged in full, compute adds 10.9 ms, and the decision lands at
35.9 ms - inside the 50 ms latency budget and a small fraction of the 250 ms
epoch.

Bottom: the same 10.9 ms of compute, magnified, split into the four stages that
make it up. Feature extraction is 93 % of it; the two model heads together are
0.54 ms.

Medians pooled over 22,671 timed windows from three recordings (Table 1).
Every value is read off the axis or off the segment it belongs to, so nothing
floats in the white space.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

# ---------------------------------------------------------------- style
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

# ---------------------------------------------------------------- content
# Table 1, pooled medians in ms.
LOOKAHEAD = 25.0
COMPUTE_T = 10.89
DECISION = LOOKAHEAD + COMPUTE_T                  # 35.89, reported as 35.9
BUDGET, EPOCH = 50.0, 250.0

# (label, ms, colour) - stages in the order they run.
PARTS = [
    ("filter, detrend, normalization", 0.17, "#8FB3E8"),
    ("feature extraction",            10.17, COMPUTE),
    ("frozen classifier",              0.40, "#17337A"),
    ("correction",                     0.14, "#6FA8FA"),
]
PARTS_T = sum(p[1] for p in PARTS)                # 10.88; medians do not add

# ---------------------------------------------------------------- layout
MM = 1 / 25.4
H = 56.0                                          # figure height, mm
fig = plt.figure(figsize=(180 * MM, H * MM))
AX_L, AX_W = 0.055, 0.915
AZ_L, AZ_W = 0.175, 0.795
ax = fig.add_axes([AX_L, 40.0 / H, AX_W, 12.5 / H])   # the 250 ms timeline
az = fig.add_axes([AZ_L, 17.0 / H, AZ_W, 6.0 / H])    # compute, magnified

XLO, XHI = -8, 262
MAG = round((AZ_W / PARTS_T) / (AX_W / (XHI - XLO)))


def brace(axes, x0, x1, y, depth, transform=None, r=2.0, lw=0.6, color=INK):
    """A curly brace over x0..x1: tips at y, point at y + depth.

    x is always data (ms). y follows `transform`: pass None for data coords
    (to brace something drawn on the bar) or ax.get_xaxis_transform() for
    axes-fraction coords (to brace something under the axis). Mixing the two
    up silently floats the brace off the axes, so it is explicit here.

    Four quarter turns joined by two flat runs - the shape of a real brace,
    rather than the square bracket matplotlib's '-[' arrowstyle would give.
    """
    xm = 0.5 * (x0 + x1)
    r = min(r, (xm - x0) / 2.2)
    h = 0.5 * depth
    verts = [
        (x0, y),
        (x0, y + h), (x0 + r, y + h),           # out of the left tip
        (xm - r, y + h),                        # flat run
        (xm, y + h), (xm, y + depth),           # into the point
        (xm, y + h), (xm + r, y + h),           # and back out
        (x1 - r, y + h),                        # flat run
        (x1, y + h), (x1, y),                   # into the right tip
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
    ]
    axes.add_patch(PathPatch(Path(verts, codes), fill=False, lw=lw,
                             edgecolor=color, clip_on=False, zorder=5,
                             transform=transform or axes.transData))

# ---------------------------------------------------------------- timeline
ax.add_patch(Rectangle((0, 0), LOOKAHEAD, 1, fc=LOOK, ec="none", zorder=3))
ax.add_patch(Rectangle((LOOKAHEAD, 0), COMPUTE_T, 1, fc=COMPUTE, ec="none",
                       zorder=3))

ax.axvline(BUDGET, color=LIMIT, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.axvline(EPOCH, color=MUTED, lw=0.9, zorder=4)

# each block is braced and named above, and carries its duration inside
brace(ax, 0, LOOKAHEAD, 1.04, 0.17)
brace(ax, LOOKAHEAD, DECISION, 1.04, 0.17)
ax.text(12.0, 1.25, "lookahead", ha="center", va="bottom", fontsize=7.2,
        color=INK, zorder=5)
ax.text(31.8, 1.25, "compute", ha="center", va="bottom", fontsize=7.2,
        color=INK, zorder=5)
ax.text(LOOKAHEAD / 2, 0.5, "25 ms", ha="center", va="center",
        fontsize=7.2, color=INK, zorder=5)
ax.text(LOOKAHEAD + COMPUTE_T / 2, 0.5, "10.9\nms", ha="center", va="center",
        fontsize=6.2, color="white", linespacing=0.95, zorder=5)

ax.set_xlim(XLO, XHI)
ax.set_ylim(-0.18, 1.55)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.spines["bottom"].set_linewidth(0.6)

ax.set_xticks([0, 25, 35.9, 50, 100, 150, 200, 250])
ax.set_xticklabels(["0", "25", "35.9", "50", "100", "150", "200", "250"])
ax.tick_params(axis="x", length=2.5, width=0.6, color=MUTED,
               labelsize=7.2, labelcolor=MUTED, pad=2)

# ------------------------------------------------- what the axis marks mean
# "decision" braces the whole 0-35.9 ms it names; the budget labels sit on
# the marks they explain.
TR = ax.get_xaxis_transform()
brace(ax, 0, DECISION, -0.32, -0.17, transform=TR)
ax.text(DECISION / 2, -0.55, "decision", transform=TR, ha="center", va="top",
        fontsize=7.2, color=INK, clip_on=False)
ax.text(BUDGET, -0.55, "50 ms latency budget", transform=TR, ha="center",
        va="top", fontsize=7.2, color=LIMIT, clip_on=False)
ax.text(140, -0.95, "time after the epoch closes  (ms)", transform=TR,
        ha="center", va="top", fontsize=7.2, color=MUTED, style="italic",
        clip_on=False)
ax.text(EPOCH, -0.95, "250 ms epoch closes", transform=TR, ha="right",
        va="top", fontsize=7.2, color=MUTED, clip_on=False)

# ---------------------------------------------------------------- the zoom
x = 0.0
centres = []
for _name, val, col in PARTS:
    az.add_patch(Rectangle((x, 0), val, 1, fc=col, ec="white", lw=0.6,
                           zorder=3))
    centres.append(x + val / 2)
    x += val

az.set_xlim(0, PARTS_T)
az.set_ylim(0, 1)
az.set_xticks([])
az.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    az.spines[s].set_visible(False)

# the only segment wide enough to hold its own label
az.text(centres[1], 0.5, "feature extraction    10.17 ms",
        ha="center", va="center", fontsize=7.4, color="white", zorder=5)

# row label, in the margin the zoom leaves free
AL = az.get_yaxis_transform()
az.text(-0.020, 0.72, "compute", transform=AL, ha="right",
        va="center", fontsize=8, color=INK, fontweight="bold", clip_on=False)
az.text(-0.020, 0.16, f"magnified ×{MAG}", transform=AL, ha="right",
        va="center", fontsize=7, color=MUTED, clip_on=False)

# the three slivers, called out below the bar
CALLOUTS = [
    (0.085,     0.0,     -0.65, "left",  "filter, detrend, normalization   0.17 ms"),
    (centres[2], centres[2], -0.65, "right", "frozen classifier   0.40 ms"),
    (centres[3], PARTS_T,    -1.60, "right", "correction   0.14 ms"),
]
for xseg, xlab, ylab, ha, text in CALLOUTS:
    az.plot([xseg, xseg], [0, ylab + 0.30], color=MUTED, lw=0.5,
            clip_on=False, zorder=2)
    az.text(xlab, ylab, text, ha=ha, va="top", fontsize=7.2, color=INK,
            clip_on=False)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure2a")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg   magnification x%d" % MAG)
