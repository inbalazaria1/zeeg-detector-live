#!/usr/bin/env python
"""
Which stretches of each night are training windows.

The NREM periods come from `hypno.py` (N1/N2/N3, gaps under 0.5 min bridged,
at least 15 min long); that rule reproduces the lab's DH3 crop to 0.1 min.  A
training window must also be at least MIN_TRAIN_MIN long, because the z-score
is seeded on the first 20 min and a shorter period is almost all seed.

The lab gave one scored crop per patient (LAB), each one NREM period:

    AO0   4    - 47.5 min
    DH3  23    - 74   min
    MF7   2    - 62   min

  donor(s)      every long NREM period of the other recordings
  own_extra(s)  s's own long periods that do not overlap its lab crop.  Later
                in the same night than the crop, so not fair as training data
                for scoring that crop.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRIAL = os.path.dirname(HERE)
ROOT = os.path.join(TRIAL, "_ext")
if os.path.join(ROOT, "cycles") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "cycles"))

import hypno as H                                                # noqa: E402

SUBJECTS = ("DH3", "AO0", "MF7")

# Recordings used only as training data, never scored.  Add a new subject here.
# ON68 is a healthy control (about 20x fewer spikes, mostly N2).  The shipped
# model was not trained on it.
EXTRA_DONORS = ("ON68",)
ALL = SUBJECTS + EXTRA_DONORS

# the lab's crops, from hypno.LAB_WINDOW
LAB = dict(H.LAB_WINDOW)

# 20 min of seed + at least 5 min to train on
MIN_TRAIN_MIN = 25.0


def _periods(subject):
    return H.cycles(subject)


def all_windows(subject):
    """Every NREM period, as `(tag, lo_min, hi_min, dur_min)`."""
    return [(f"c{p['period']}", p["start_min"], p["end_min"], p["dur_min"])
            for p in _periods(subject)]


def long_windows(subject):
    """The NREM periods long enough to get past the 20 min seed."""
    return [w for w in all_windows(subject) if w[3] >= MIN_TRAIN_MIN]


def _overlaps(a, b, lab):
    return a < lab[1] and b > lab[0]


def own_extra(subject):
    """`subject`'s own long NREM periods that do not touch its lab crop."""
    lab = LAB[subject]
    return [w for w in long_windows(subject) if not _overlaps(w[1], w[2], lab)]


def donor(subject):
    """`[(donor_subject, tag, lo, hi, dur)]`: the other recordings' long periods."""
    return [(s,) + w for s in SUBJECTS if s != subject for w in long_windows(s)]


def lab_window(subject):
    return ("lab",) + LAB[subject] + (LAB[subject][1] - LAB[subject][0],)


def cells(subject):
    """Every (tag, lo, hi) one subject needs a matrix for."""
    out = [lab_window(subject)]
    out += [w for w in long_windows(subject)]
    return out


def main():
    print(f"\nNREM periods per recording; a training window needs "
          f">= {MIN_TRAIN_MIN:g} min\n")
    tot_all = tot_use = 0.0
    for s in ALL:
        lab = LAB.get(s)
        print(f"{s}   " + (f"lab window {lab[0]:g} - {lab[1]:g} min "
                           f"({lab[1] - lab[0]:g} min)" if lab
                           else "donor only, no lab window"))
        for tag, lo, hi, dur in all_windows(s):
            use = dur >= MIN_TRAIN_MIN
            ov = lab is not None and _overlaps(lo, hi, lab)
            tot_all += dur
            tot_use += dur if use else 0.0
            print(f"    {tag:>3s}  {lo:6.1f} - {hi:6.1f} min  {dur:5.1f} min"
                  f"   {'train' if use else 'TOO SHORT':9s}"
                  f"   {'overlaps the lab window' if ov else ''}")
        print(f"    donor pool for the others:    "
              f"{sum(w[3] for w in long_windows(s)):.1f} min")
        if lab:
            print(f"    own extra (tier 3):           "
                  f"{sum(w[3] for w in own_extra(s)):.1f} min")
        print()
    print(f"total NREM {tot_all:.1f} min, usable as training {tot_use:.1f} min, "
          f"lab windows {sum(LAB[s][1] - LAB[s][0] for s in SUBJECTS):.1f} min")
    print("\nper fold, what the correction may be fitted on (tier 1):")
    for s in SUBJECTS:
        d = donor(s)
        e = sum(w[3] for x in EXTRA_DONORS for w in long_windows(x))
        old = sum(LAB[o][1] - LAB[o][0] for o in SUBJECTS if o != s)
        print(f"  test {s}: {sum(w[4] for w in d):6.1f} min of donor NREM "
              f"in {len(d)} windows (+{e:.1f} min if ON68 joins), "
              f"against {old:.1f} min today "
              f"({sum(w[4] for w in d) / old:.1f}x)")


if __name__ == "__main__":
    main()
