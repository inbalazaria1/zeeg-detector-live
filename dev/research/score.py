#!/usr/bin/env python
"""
Exact-epoch scoring.

One-to-one greedy match: candidate pairs sorted by (distance, i, j), each index
used once, and `detection - reference` must fall in an offset range.

    exact   det - ref in [-0.001, +0.001] s, the default and the only metric
            we report.  Times are multiples of 0.25 s, so the 1 ms is float
            slack: the detection has to land on the same 250 ms epoch.

`next`, `pm025` and `pm05` are looser rules, kept only to cross-check older
numbers.  `next` is [0, +250 ms], not symmetric, because a causal chain lags
the offline one and never fires early.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "_ext")      # the copied research modules
for _p in (os.path.join(ROOT, "bin20"),
           os.path.join(ROOT, "simulator-without-cleaning", "rt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as P                                             # noqa: E402

EPS = 0.001
WINDOWS = {
    "exact": (-EPS, EPS),
    "next":  (-EPS, 0.25 + EPS),
    "pm025": (-0.25 - EPS, 0.25 + EPS),
    "pm05":  (-0.5 - EPS, 0.5 + EPS),
}
DEFAULT_WINDOW = "exact"

THS = np.round(np.arange(0.30, 0.981, 0.01), 2)
REPORT_THS = (0.70, 0.74)

LABEL = {"exact": "exact epoch",
         "next":  "this epoch or the next",
         "pm025": "+-1 epoch",
         "pm05":  "+-0.5 s (bin20's rule)"}


def match_window(ref, det, lo, hi):
    """One-to-one greedy match where `det - ref` must lie in [lo, hi].

    `pipeline.match` with an offset range instead of a symmetric tolerance.
    """
    a = np.asarray(ref, dtype=float)
    b = np.asarray(det, dtype=float)
    if not len(a) or not len(b):
        return []
    first = np.searchsorted(b, a + lo, "left")
    last = np.searchsorted(b, a + hi, "right")
    cands = sorted((abs(a[i] - b[j]), i, j)
                   for i in range(len(a)) for j in range(first[i], last[i]))
    ua, ub, pairs = set(), set(), []
    for _, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pairs.append((i, j))
    return pairs


def counts(times, ref, lo, th, window=DEFAULT_WINDOW):
    """TP/FP/FN of one threshold, both lists restricted to t >= lo."""
    t = [x for x in times if x >= lo]
    keep = [x for x in ref if x >= lo]
    off_lo, off_hi = WINDOWS[window]
    tp = len(match_window(keep, t, off_lo, off_hi))
    return dict(th=float(th), det=len(t), TP=tp, FN=len(keep) - tp,
                FP=len(t) - tp, n_ref=len(keep),
                recall=tp / len(keep) if keep else 0.0,
                prec=tp / len(t) if t else 0.0,
                f1=2 * tp / (len(keep) + len(t)) if len(keep) + len(t) else 0.0)


def curve(proba, ref, lo, window=DEFAULT_WINDOW, ths=THS):
    """One row per threshold in 0.30 .. 0.98 step 0.01."""
    return [counts(P.detect(proba, threshold=float(th))[0], ref, lo, th, window)
            for th in ths]


# ----------------------------------------------------------------------
# accessors
# ----------------------------------------------------------------------
def at(rows, th):
    """The row at threshold `th`, which must be on the grid."""
    return next(r for r in rows if abs(r["th"] - th) < 1e-9)


def at070(rows):
    return at(rows, 0.70)


def at074(rows):
    return at(rows, 0.74)


def oracle(rows):
    """Best F1 over the grid.  Context only: picking it uses the answer."""
    return max(rows, key=lambda r: r["f1"])
