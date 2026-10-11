#!/usr/bin/env python
"""
Deliverable B of `DATA_EXPORT_BRIEF.md`: the threshold sweep counts.

Nothing here is new science.  The three model variants are the three rows of
`dev/RESULTS.md`'s headline table, and the corrected probability vectors they
produce are the ones the research package already saved:

    frozen                      det.npz     frozen__<S>            alpha 0
    correction_crops            nrem_A.npz  lab__hold70__<S>       the published
                                            `lab` row - the 5-SEED BAG, seed set
                                            A, which is what RESULTS.md quotes
                                            (mean p 0.382 / F1 0.346).  det.json's
                                            deterministic `lab` is a different
                                            number (0.374 / 0.337) and is not the
                                            published one.
    correction_nrem_seeddrop    det.npz     nrem_ns__hold70__<S>   the shipped row

`alpha` is baked into those vectors - each is already
`sigmoid(logit(frozen) + alpha * correction)` at that fold's `hold70` alpha - so
holding alpha fixed while sweeping the threshold is automatic here, which is
exactly what the brief asks for.

The decision rule applied at each threshold is the whole of it and is the
research package's own code, not a re-implementation:

    score.curve -> pipeline.detect (threshold, 2 s refractory, 20 min bins,
                   short-final-bin drop)
                -> score.counts    (restrict both lists to t >= 1200 s, then the
                   one-to-one greedy EXACT-epoch match)

    RESEARCH_PACKAGE=/path/to/research-package ./.venv/bin/python export_threshold_sweep.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The research package is not distributed; point RESEARCH_PACKAGE at your copy.
PKG = os.environ.get("RESEARCH_PACKAGE")
if not PKG:
    sys.exit("set RESEARCH_PACKAGE to the research package directory")
SLEEP = os.path.join(PKG, "sleep")
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "threshold_sweep.csv"))
OUT2 = os.path.abspath(os.path.join(HERE, "..", "data",
                                    "threshold_sweep_published_pr.csv"))

sys.path.insert(0, PKG)
import score as SC                                              # noqa: E402

NORM_SECONDS = 1200.0
SUBJECTS = ("DH3", "AO0", "MF7")
THS = np.round(np.arange(0.300, 0.9901, 0.005), 3)

# (variant id, npz file, key template).  `None` template = the frozen vector.
VARIANTS = (
    ("frozen", "det.npz", "frozen__{s}"),
    ("correction_crops", "nrem_A.npz", "lab__hold70__{s}"),
    ("correction_nrem_seeddrop", "det.npz", "nrem_ns__hold70__{s}"),
)
# the alpha each of those vectors was built with, read back from the json so it
# is the recorded value and not a value typed in here
ALPHA_SRC = {"frozen": (None, None, None),
             "correction_crops": ("nrem_A.json", "lab", "hold70"),
             "correction_nrem_seeddrop": ("det.json", "nrem_ns", "hold70"),
             "correction_crops_expandg10_joint": ("g_bagA.json", "lab", "joint")}

# `DATA_EXPORT_BRIEF.md` is self-inconsistent about what `correction_crops` is:
# its 5.2 numbers (mean p 0.382 / F1 0.346) are the `expand20` lab correction
# picked by `hold70`, its 5.4 numbers (0.415 / 0.384 / MF7 0.514) are the
# `expandg10` lab correction picked by `joint` - a different training cell AND a
# different rule, the one that lets alpha and the threshold move together.  The
# brief's own alpha instruction ("hold alpha, do not re-select") selects the
# first, so that is what `correction_crops` is in the main file; the second is
# exported here beside it so RESULTS.md 9's PR table stays reproducible.
EXTRA = (("correction_crops_expandg10_joint", "g_bagA.npz", "lab__joint__{s}"),)


def alphas(variant):
    import json
    f, kind, pick = ALPHA_SRC[variant]
    if f is None:
        return {s: 0.0 for s in SUBJECTS}
    with open(os.path.join(SLEEP, f)) as fh:
        res = json.load(fh)["res"][kind]
    return {s: float(res[s][pick]["alpha"]) for s in SUBJECTS}


def sweep(variants, npz):
    rows = []
    for variant, fname, tmpl in variants:
        a = alphas(variant)
        for s in SUBJECTS:
            p = np.asarray(npz[fname][tmpl.format(s=s)], dtype=float)
            ref = np.asarray(npz[fname][f"ref__{s}"], dtype=float)
            curve = SC.curve(p, ref, NORM_SECONDS, window="exact", ths=THS)
            for r in curve:
                rows.append(dict(variant=variant, recording=s, alpha=a[s],
                                 threshold=f"{r['th']:.3f}", n_reference=r["n_ref"],
                                 n_detected=r["det"], tp=r["TP"], fp=r["FP"],
                                 fn=r["FN"]))
            print(f"  {variant:34s} {s}  {len(p)} epochs, alpha {a[s]:.2f}, "
                  f"{r['n_ref']} scored reference detections", flush=True)
    return rows


def write(rows, path):
    cols = ["variant", "recording", "alpha", "threshold", "n_reference",
            "n_detected", "tp", "fp", "fn"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {path}\n")


def main():
    files = {v[1] for v in VARIANTS} | {v[1] for v in EXTRA}
    npz = {f: np.load(os.path.join(SLEEP, f), allow_pickle=True) for f in files}
    write(sweep(VARIANTS, npz), OUT)
    write(sweep(EXTRA, npz), OUT2)


if __name__ == "__main__":
    main()
