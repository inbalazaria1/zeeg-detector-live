"""
Figure - Threshold sweep for the corrected detector, averaged over recordings.

Precision, recall and F1 as the decision threshold moves from 0.66 to 0.80.
Precision is flat from 0.70 onward - raising the threshold buys no more of it -
while recall falls away steeply, so mean F1 peaks at 0.70 and only loses from
there. That is the operating point the system holds, and the column reported in
Table 2 (0.388 / 0.392 / 0.383).

Mean over three recordings, exact-epoch matching.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

INK, MUTED, GRID = "#1F2328", "#6B7280", "#E3E6EA"
F1_C, PREC_C, REC_C = "#1E40AF", "#60A5FA", "#C2410C"
ACCENT, BAND = "#2563EB", "#EAF1FD"

# ------------------------------------------------------------------ data
TH = [0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
PREC = [0.332, 0.361, 0.388, 0.388, 0.392, 0.391, 0.393, 0.385]
REC = [0.403, 0.402, 0.392, 0.334, 0.280, 0.216, 0.157, 0.107]
F1 = [0.355, 0.372, 0.383, 0.353, 0.323, 0.276, 0.223, 0.164]

HELD = 0.70                                   # the operating point
H = TH.index(HELD)
assert F1[H] == max(F1), "the marked point must be the F1 peak"

# ------------------------------------------------------------------ layout
MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 84 * MM))
ax = fig.add_axes([0.070, 0.165, 0.845, 0.750])

XLO, XHI, YHI = 0.649, 0.826, 0.46

# the held threshold, banded behind everything
ax.axvspan(HELD - 0.006, HELD + 0.006, color=BAND, zorder=1)
ax.axvline(HELD, color=ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=2)

SERIES = [
    ("mean precision", PREC, PREC_C, 1.1, 3.8),
    ("mean recall",    REC,  REC_C,  1.1, 3.8),
    ("mean F1",        F1,   F1_C,   1.9, 4.6),
]
for name, ys, col, lw, ms in SERIES:
    ax.plot(TH, ys, color=col, lw=lw, marker="o", ms=ms, mec="white", mew=0.6,
            solid_capstyle="round", zorder=4, clip_on=False)
    ax.text(TH[-1] + 0.004, ys[-1], name, ha="left", va="center", fontsize=8,
            color=col, fontweight="bold" if name == "mean F1" else "normal",
            clip_on=False)

# ------------------------------------------------------- mark the best point
ax.plot([HELD], [F1[H]], marker="o", ms=10.5, mfc="none", mec=ACCENT, mew=1.4,
        zorder=5, clip_on=False)
ax.annotate("best mean F1   0.383\nthreshold held at 0.70",
            xy=(HELD, F1[H] + 0.014), xytext=(0.694, 0.437),
            ha="center", va="bottom", fontsize=8, color=ACCENT,
            linespacing=1.25,
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7,
                            shrinkA=2, shrinkB=1))

# the two facts that make 0.70 the peak, read off the curves they belong to
ax.text(0.766, 0.405, "precision is flat past 0.70", ha="center", va="bottom",
        fontsize=7.4, color=PREC_C)
ax.text(0.792, 0.135, "recall collapses", ha="right", va="top", fontsize=7.4,
        color=REC_C)

# ------------------------------------------------------------------ axes
ax.set_xlim(XLO, XHI)
ax.set_ylim(0.06, YHI)
ax.set_xticks(TH)
ax.set_xticklabels([f"{t:.2f}" for t in TH])
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.set_xlabel("decision threshold", fontsize=8, color=MUTED, labelpad=4)
ax.set_ylabel("agreement with the offline pipeline", fontsize=8, color=MUTED,
              labelpad=5)
ax.tick_params(axis="both", length=2.5, width=0.6, color=MUTED,
               labelsize=7.4, labelcolor=MUTED, pad=2)
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
    ax.spines[s].set_linewidth(0.6)

# the held tick, called out in the accent so the axis agrees with the marker
ax.get_xticklabels()[H].set_color(ACCENT)
ax.get_xticklabels()[H].set_fontweight("bold")

fig.text(0.070, 0.022, "mean over three recordings, exact-epoch matching",
         ha="left", va="bottom", fontsize=7.2, color=MUTED, style="italic")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_threshold")
fig.savefig(out + ".png", dpi=600, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
print("wrote", out + ".png / .svg")
