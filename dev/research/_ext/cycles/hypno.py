#!/usr/bin/env python
"""
The subject table, and the NREM periods of each subject's hypnogram.

A hypnogram is a text file of whitespace-separated stage codes, one per
`epoch_sec` seconds: 0 W, 1 N1, 2 N2, 3 N3, 4 REM, -1 unscored.  The expert
scorings in `scoring-sleep_stages/` are one code per second.

"Cycle k" below means the k-th NREM period, not a textbook NREM+REM cycle:
some nights have almost no REM, and spikes are counted in NREM anyway.

The rule: NREM is N1/N2/N3, a wake/REM/unscored gap shorter than `bridge_min`
does not split a period, and a period needs at least `min_nrem_min`.  The
defaults (0.5 and 15 min) reproduce the lab's own DH3 crop: 23.1-74.3 min
against their 23-74.  `--sweep` shows how other rules do.

AO0 and EA3001 are the same recording (the AO0-mff folder).  AO0 uses the
lab's expert scoring, EA3001 a YASA scoring, so they are one patient, never
two.  AO0.txt covers only 312.5 of the 398.9 min, but its first two NREM
periods line up with the lab's 4-47.5 min crop, and the lab decided to use it.

From dev/research:

    ../../.venv/bin/python _ext/cycles/hypno.py
    ../../.venv/bin/python _ext/cycles/hypno.py --sweep
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCORING = os.path.join(ROOT, "scoring-sleep_stages")
YASA = os.path.join(ROOT, "sleepeegpy", "staging_output")

WAKE, N1, N2, N3, REM, UNSCORED = 0, 1, 2, 3, 4, -1
NREM_CODES = (N1, N2, N3)
STAGE_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "x"}

# the default period rule, see the module docstring
BRIDGE_MIN = 0.5
MIN_NREM_MIN = 15.0

# One entry per subject.  To add one, copy a row:
#   rec        name of the recording; subjects on the same .mff share it,
#              and with it the raw cache
#   mff        the .mff folder, relative to _ext/
#   hypno      the hypnogram file
#   epoch_sec  seconds per stage code in the hypnogram
#   source     who scored it, only printed
#   rec_min    recording length in minutes
#   max_min    how far it is safe to read; normally rec_min
#
# max_min is lower only if the .mff has a gap (more than one epoch in
# epochs.xml).  DH3 has a 4.6 min gap at minute 274.7, and mff_direct's block
# arithmetic goes wrong past it, so DH3 stops at 274.6.  All its NREM periods
# end by minute 180, so nothing is lost.
SUBJECTS = {
    "DH3": dict(rec="DH3", mff="mff's/DH3_20220529_120959.mff",
                hypno=os.path.join(SCORING, "DH3.txt"), epoch_sec=1,
                source="expert", rec_min=316.87, max_min=274.6),
    "MF7": dict(rec="MF7", mff="mff's/MF7_20230225_110959.mff",
                hypno=os.path.join(SCORING, "MF7.txt"), epoch_sec=1,
                source="expert", rec_min=366.20, max_min=366.20),
    "ON68": dict(rec="ON68", mff="mff's/ON68_Sleep_234420.mff",
                 hypno=os.path.join(SCORING, "ON68.txt"), epoch_sec=1,
                 source="expert", rec_min=397.80, max_min=397.80, healthy=True),
    "AO0": dict(rec="EA3001", mff="mff's/AO0-mff",
                hypno=os.path.join(SCORING, "AO0.txt"), epoch_sec=1,
                source="expert scoring, assigned to this recording by the lab",
                rec_min=398.93, max_min=398.93),
    # the same recording as AO0, windowed by YASA instead of by AO0.txt
    "EA3001": dict(rec="EA3001", mff="mff's/AO0-mff",
                   hypno=os.path.join(YASA, "EA3001_hypnogram.txt"), epoch_sec=30,
                   source="YASA", rec_min=398.93, max_min=398.93,
                   same_recording_as="AO0"),
}

# the lab's hand-picked crops, to check the period rule against
LAB_WINDOW = {"AO0": (4.0, 47.5), "DH3": (23.0, 74.0), "MF7": (2.0, 62.0)}


def load(name):
    """(stages, epoch_sec) for one row of SUBJECTS."""
    cfg = SUBJECTS[name]
    with open(cfg["hypno"]) as fh:
        return np.array([int(x) for x in fh.read().split()]), cfg["epoch_sec"]


def _runs(mask):
    """[(start, stop)] of the True runs of a boolean array, stop exclusive."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _bridge(mask, max_gap):
    """Fill False runs shorter than `max_gap` that have True on both sides."""
    out = mask.copy()
    if max_gap <= 0:
        return out
    for a, b in _runs(~mask):
        if a > 0 and b < len(mask) and b - a < max_gap:
            out[a:b] = True
    return out


def nrem_periods(stages, epoch_sec=1, bridge_min=BRIDGE_MIN,
                 min_nrem_min=MIN_NREM_MIN, max_min=None):
    """The NREM periods of a hypnogram, in minutes from the start.

    Periods past `max_min` are dropped, and one that crosses it is cut short.
    """
    per_min = 60.0 / epoch_sec
    is_nrem = np.isin(stages, NREM_CODES)
    bridged = _bridge(is_nrem, int(round(bridge_min * per_min)))

    out = []
    for a, b in _runs(bridged):
        if (b - a) < min_nrem_min * per_min:
            continue
        lo, hi = float(a) / per_min, float(b) / per_min
        if max_min is not None:
            hi = min(hi, max_min)
            if hi - lo < min_nrem_min:
                continue
        seg = stages[a:int(round(hi * per_min))]
        out.append({
            "start_min": float(lo), "end_min": float(hi),
            "dur_min": float(hi - lo),
            **{f"{STAGE_NAME[s]}_min": round(float((seg == s).sum()) / per_min, 2)
               for s in (N1, N2, N3, REM, WAKE, UNSCORED)},
        })
    for i, p in enumerate(out, 1):
        p["period"] = i
    return out


def cycles(name, bridge_min=BRIDGE_MIN, min_nrem_min=MIN_NREM_MIN):
    """The NREM periods of one subject, stopping at its max_min."""
    stages, es = load(name)
    return nrem_periods(stages, es, bridge_min, min_nrem_min,
                        SUBJECTS[name].get("max_min"))


def describe(name, periods, n=None):
    cfg = SUBJECTS[name]
    lines = [f"{name}  ({cfg['source']}; recording {cfg['rec_min']:.1f} min"
             + (f", usable to {cfg['max_min']:.1f} min"
                if cfg.get("max_min", 1e9) < cfg["rec_min"] - 0.05 else "")
             + f"): {len(periods)} NREM periods"]
    for p in periods[:n]:
        lines.append(
            f"   cycle {p['period']}  {p['start_min']:6.1f} - {p['end_min']:6.1f} min "
            f"({p['dur_min']:5.1f} min)   N1 {p['N1_min']:5.1f}  N2 {p['N2_min']:5.1f}  "
            f"N3 {p['N3_min']:5.1f}  W {p['W_min']:4.1f}")
    if n and len(periods) > n:
        lines.append(f"   ... {len(periods) - n} more")
    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="NREM periods of every recording")
    ap.add_argument("--bridge-min", type=float, default=BRIDGE_MIN)
    ap.add_argument("--min-nrem-min", type=float, default=MIN_NREM_MIN)
    ap.add_argument("--n", type=int, default=4, help="periods to print")
    ap.add_argument("--sweep", action="store_true",
                    help="show how the period count moves with the rule")
    a = ap.parse_args()

    if a.sweep:
        print("\nhow many NREM periods each rule finds, and where cycle 1 ends\n")
        print(f"{'bridge':>7s}{'minlen':>7s}  " +
              "".join(f"{s:>22s}" for s in SUBJECTS))
        for br in (0.5, 1.0, 2.0, 3.0):
            for mn in (15.0, 20.0):
                row = ""
                for s in SUBJECTS:
                    p = cycles(s, br, mn)
                    row += f"{len(p):>4d} " + (
                        f"c1 {p[0]['start_min']:.1f}-{p[0]['end_min']:.1f}".rjust(17)
                        if p else "".rjust(17))
                print(f"{br:>7g}{mn:>7g}  " + row)
        print("\nlab trim table: " + ", ".join(
            f"{k} {v[0]:g}-{v[1]:g}" for k, v in LAB_WINDOW.items()))
        return

    print(f"\nNREM = N1/N2/N3; intrusions < {a.bridge_min:g} min bridged; "
          f"period >= {a.min_nrem_min:g} min\n")
    for name in SUBJECTS:
        per = cycles(name, a.bridge_min, a.min_nrem_min)
        print(describe(name, per, a.n))
        if name in LAB_WINDOW:
            lo, hi = LAB_WINDOW[name]
            print(f"   (lab trim table: {lo:g} - {hi:g} min)")
        if SUBJECTS[name].get("same_recording_as"):
            print(f"   (same recording as "
                  f"{SUBJECTS[name]['same_recording_as']}, different scoring)")
        print()


if __name__ == "__main__":
    main()
