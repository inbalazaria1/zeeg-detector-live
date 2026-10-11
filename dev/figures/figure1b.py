"""
Figure 1b - Offline pipeline vs. causal replacement, stage by stage.

For each processing stage, how much signal is read into the past and into the
future relative to the sample currently being processed.

Both panels use a LINEAR time axis, so every bar is drawn in true proportion.
The overview spans +/-17.5 s; the detail panel magnifies the region around
t = 0, where the causal chain's 25 ms lookahead and the 250 ms epoch live.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK       = "#1F2328"
MUTED     = "#6B7280"
OFFLINE   = "#6B7280"
CAUSAL    = "#2563EB"
UNCHANGED = "#A5AAB2"
FUTURE    = "#C2410C"

BAR_LW = 5.2
BIG    = 1e6            # stands in for "whole recording"
WHOLE  = "whole"

# ---------------------------------------------------------------- content
# (stage, [(row, past, future, colour)])
GROUPS = [
    ("band-pass", [
        ("offline", 16.5,  16.5,  OFFLINE),
        ("causal",  WHOLE, 0.025, CAUSAL),
    ]),
    ("detrend", [
        ("offline", 1.0,   1.0,   OFFLINE),
        ("causal",  2.0,   0.0,   CAUSAL),
    ]),
    ("normalization", [
        ("offline", WHOLE, WHOLE, OFFLINE),
        ("causal",  WHOLE, 0.0,   CAUSAL),
    ]),
    ("classifier", [
        ("unchanged", 0.25, 0.0,  UNCHANGED),
    ]),
    ("detection", [
        ("unchanged", 2.0,  0.0,  UNCHANGED),
    ]),
]

SPANS = ["±16.5 s", "+25 ms",
         "±1 s", "2 s trailing",
         "whole recording", "expanding",
         "250 ms epoch", "2 s refractory"]

# ---------------------------------------------------------------- layout
ROW_GAP, GRP_GAP, HEAD = 1.0, 1.5, 1.05
y, rows, heads = 0.0, [], []
for gname, members in GROUPS:
    y -= HEAD
    heads.append((y, gname))
    for m in members:
        y -= ROW_GAP
        rows.append((y,) + m)
    y -= GRP_GAP - ROW_GAP
YMIN, YTOP = y - 0.45, 1.55

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 90 * MM))
ax_main = fig.add_axes([0.145, 0.160, 0.425, 0.755])
ax_zoom = fig.add_axes([0.605, 0.160, 0.222, 0.755])

MAIN_LO, MAIN_HI = -17.5, 17.5
ZOOM_LO, ZOOM_HI = -0.26, 0.05


def edges(past, future):
    return (-BIG if past == WHOLE else -past,
            BIG if future == WHOLE else future)


def draw_panel(ax, xlo, xhi, arrows=True):
    span = xhi - xlo
    ax.axvspan(0, xhi, color=FUTURE, alpha=0.07, lw=0, zorder=0)
    ax.add_line(Line2D([0, 0], [YMIN, 1.05], color=INK, lw=0.7,
                       ls=(0, (1.6, 1.6)), zorder=4))
    for (yy, _lab, past, future, colour) in rows:
        x0, x1 = edges(past, future)
        c0, c1 = max(x0, xlo), min(x1, xhi)
        if c1 < c0:
            continue
        ax.add_line(Line2D([c0, c1], [yy, yy], lw=BAR_LW, color=colour,
                           solid_capstyle="butt", zorder=3))
        if arrows and x0 < xlo:
            ax.plot(xlo - 0.012 * span, yy, marker="<", ms=5.0, color=colour,
                    mec="none", clip_on=False, zorder=5)
        if arrows and x1 > xhi:
            ax.plot(xhi + 0.012 * span, yy, marker=">", ms=5.0, color=colour,
                    mec="none", clip_on=False, zorder=5)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(YMIN, YTOP)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=2.5, width=0.6, color=MUTED,
                   labelsize=7.2, labelcolor=MUTED, pad=2)


draw_panel(ax_main, MAIN_LO, MAIN_HI)
draw_panel(ax_zoom, ZOOM_LO, ZOOM_HI, arrows=False)

# ---------------------------------------------------------------- ticks
ax_main.set_xticks([-15, -10, -5, 0, 5, 10, 15])
ax_main.set_xticklabels(["−15 s", "−10 s", "−5 s", "0", "5 s", "10 s", "15 s"])
ax_zoom.set_xticks([-0.25, -0.125, 0])
ax_zoom.set_xticklabels(["−250 ms", "−125 ms", "0"])

# ---------------------------------------------------------------- labels
LAB = ax_main.get_yaxis_transform()
for (yy, lab, *_rest) in rows:
    ax_main.text(-0.150, yy, lab, transform=LAB, ha="left", va="center",
                 fontsize=8, color=MUTED, clip_on=False)
for yy, gname in heads:
    ax_main.text(-0.215, yy, gname, transform=LAB, ha="left", va="center",
                 fontsize=9, color=INK, fontweight="bold", clip_on=False)

ZLAB = ax_zoom.get_yaxis_transform()
for (yy, *_), span in zip(rows, SPANS):
    ax_zoom.text(1.075, yy, span, transform=ZLAB, ha="left", va="center",
                 fontsize=7.6, color=INK, clip_on=False)

# ---------------------------------------------------------------- headers
ax_main.text(MAIN_LO * 0.50, 0.34, "past — available", ha="center",
             va="bottom", fontsize=8, color=MUTED, style="italic")
ax_main.text(MAIN_HI * 0.50, 0.34, "future — unavailable", ha="center",
             va="bottom", fontsize=8, color=FUTURE, style="italic")
ax_zoom.text(ZOOM_LO, YTOP - 0.04, "detail at t = 0   (×56)",
             ha="left", va="top", fontsize=7.4, color=MUTED)

# the one place the causal chain crosses zero - only legible magnified
ax_zoom.annotate("25 ms", xy=(0.0125, rows[1][0] - 0.20),
                 xytext=(0.032, rows[1][0] - 0.95),
                 ha="center", va="top", fontsize=7.4, color=CAUSAL,
                 arrowprops=dict(arrowstyle="-", color=CAUSAL, lw=0.6,
                                 shrinkA=0.5, shrinkB=1.5))

ax_main.set_xlabel("time read relative to the present sample   (true scale)",
                   fontsize=7.8, color=MUTED, labelpad=4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure1b")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
