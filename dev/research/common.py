#!/usr/bin/env python
"""
Model 1's probability for a matrix, and the scoring rules shared with it.

The build only uses `proba`. The rest scores the three lab crops and needs
research-tree caches that are not included here.

* Epoch i is the 250 ms starting at `i * 0.25 s` from the start of the crop.
* Scoring starts at second 1200: the 20 min calibration block is never scored.
* `ref` is not an annotation. It is the offline chain's own detections
  (33001-tap zero-phase band-pass, centred 2 s SMA, whole-crop z-score,
  threshold 0.70, 2 s refractory), so every score is agreement with that chain.
* `ref` has been through the refractory, so most epochs within 2 s of a hit
  are missing from it even if the teacher scored them high. `soft_targets` is
  the teacher before that.
"""
import os
import sys
import functools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "_ext")
for _p in (HERE,
           os.path.join(ROOT, "bin20"),
           os.path.join(ROOT, "simulator-without-cleaning", "rt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as P                                             # noqa: E402
import subjects as S                                             # noqa: E402
import feats as F                                                # noqa: E402
import score as SC                                               # noqa: E402

SUBJECTS = ("DH3", "AO0", "MF7")

BASE_CHAIN = "b25_o3"        # butter-o3 0.1 Hz HP + 51-tap 70 Hz LP read 25 late
BASE_NORM = "expand20"       # expanding z-score seeded on the 20 min block
CALIB_SEC = 1200.0           # scoring starts here, always

# the frozen model's exact-epoch (precision, F1) on the lab crops; `gate`
# checks a setup still reproduces them
BASELINE = {
    0.70: {"DH3": (0.306, 0.334), "AO0": (0.234, 0.249), "MF7": (0.408, 0.434)},
    0.74: {"DH3": (0.370, 0.303), "AO0": (0.218, 0.212), "MF7": (0.455, 0.453)},
}
BASELINE_MEAN_F1 = {0.70: 0.339, 0.74: 0.323}


# ======================================================================
# epochs, references, labels
# ======================================================================
def epoch_times(n):
    """Epoch start times in seconds from the start of the crop."""
    return np.round(np.arange(n) * P.EPOCH_SEC, 2)


_CONFIGURED = None


def reference(subject):
    """The offline chain's spike times over the subject's lab crop.

    Needs research-tree caches not included here. Also repoints `pipeline` at
    this subject, so call it again whenever a loop comes back to a subject.
    """
    global _CONFIGURED
    ref = list(S.configure(subject))
    _CONFIGURED = subject
    return ref


def scored_reference(subject):
    """`reference` from second 1200 on."""
    return [t for t in reference(subject) if t >= CALIB_SEC]


def hard_labels(times, ref):
    """1 on every epoch the offline chain fired on, 0 elsewhere.

    Exact match on the rounded time, the same rule as the metric. A tolerance
    would make the training target looser than the score.
    """
    want = set(np.round(np.asarray(ref, dtype=float), 2).tolist())
    return np.fromiter((float(t) in want for t in times), dtype=np.int8,
                       count=len(times))


# ======================================================================
# design matrices
# ======================================================================
def cell(subject, chain=BASE_CHAIN, norm=BASE_NORM):
    """`(x, times)` for one lab crop. Needs research-tree caches not included here.

    Builds the matrix if it is not cached: 2-3 min on 7 cores, so one at a time.
    """
    x = F.build_x(subject, chain, norm)
    return x, epoch_times(len(x))


def teacher_x(subject):
    """The offline chain's matrix for a lab crop: 16.5 s lookahead, global z-score.

    Needs research-tree caches not included here. Model 1 on this, thresholded
    at 0.70, gives `reference(subject)`.
    """
    p = os.path.join(ROOT, "bin20", "cache", subject, "offline__global.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_pickle(p)


@functools.lru_cache(maxsize=None)
def soft_targets(subject):
    """The teacher's per-epoch probability, before the refractory."""
    return P.model().predict_proba(teacher_x(subject))[:, 1]


def proba(x):
    """Model 1's spike probability for each row of a matrix."""
    return P.model().predict_proba(x[P.model().get_booster().feature_names])[:, 1]


# ======================================================================
# scoring
# ======================================================================
def curve(p, ref, lo=CALIB_SEC, ths=SC.THS):
    """Exact-epoch threshold curve, scored from `lo` on."""
    return SC.curve(p, ref, lo, window="exact", ths=ths)


def at(rows, th):
    return SC.at(rows, th)


def oracle(rows):
    return SC.oracle(rows)


def mean_row(curves, th):
    """`(mean precision, mean F1)` at `th` over a dict of subject -> curve."""
    r = [SC.at(c, th) for c in curves.values()]
    return float(np.mean([x["prec"] for x in r])), float(np.mean([x["f1"] for x in r]))


def table(curves, title, ths=(0.70, 0.74), baseline=True):
    """Print per-subject precision / F1 at each threshold, then the mean."""
    subs = [s for s in SUBJECTS if s in curves]
    print(f"\n=== {title} - exact epoch ===")
    print(f"{'th':>5} " + " ".join(f"{s + ' p/F1':>15}" for s in subs)
          + f" {'mean p':>7} {'mean F1':>8} {'d vs base':>10}")
    print("-" * (6 + 16 * len(subs) + 28))
    for th in ths:
        r = [SC.at(curves[s], th) for s in subs]
        mp = float(np.mean([x["prec"] for x in r]))
        mf = float(np.mean([x["f1"] for x in r]))
        d = (f"{mf - BASELINE_MEAN_F1[th]:+10.3f}"
             if baseline and th in BASELINE_MEAN_F1 else " " * 10)
        print(f"{th:>5.2f} "
              + " ".join(f"{x['prec']:.3f} / {x['f1']:.3f}".rjust(15) for x in r)
              + f" {mp:>7.3f} {mf:>8.3f}{d}")
    b = [SC.oracle(curves[s]) for s in subs]
    print(f"{'best':>5} "
          + " ".join(f"{x['f1']:.3f} @{x['th']:.2f}".rjust(15) for x in b)
          + f" {'':>7} {np.mean([x['f1'] for x in b]):>8.3f}")


def gate():
    """Check the frozen model still scores `BASELINE` on the lab crops.

    Needs research-tree caches not included here.
    """
    ok = True
    for s in SUBJECTS:
        ref = reference(s)
        x, _ = cell(s)
        c = curve(proba(x), ref)
        for th, want in BASELINE.items():
            r = SC.at(c, th)
            got = (round(r["prec"], 3), round(r["f1"], 3))
            if got != want[s]:
                print(f"  GATE FAIL {s} th{th}: got {got}, want {want[s]}")
                ok = False
    print("  gate: " + ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    gate()
