#!/usr/bin/env python
"""
§5 of `DATA_EXPORT_BRIEF.md`, checked against the two exported CSVs.

Prints every expected value next to the obtained one and a PASS / FAIL.  It
computes precision / recall / F1 from the exported counts only - nothing is read
back out of the research package - so a PASS here means the CSV a figure will be
drawn from carries the published numbers, not that some other object does.
"""
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
SUBJECTS = ("DH3", "AO0", "MF7")

ok_all = True


def mark(got, want, tol, label, fmt="{:.3f}"):
    global ok_all
    ok = got is not None and abs(got - want) <= tol
    ok_all &= ok
    g = "unreachable" if got is None else fmt.format(got)
    print(f"  {label:52s} expected {fmt.format(want):>8s}   got {g:>11s}   "
          f"{'PASS' if ok else 'FAIL'}")


def mark_eq(got, want, label):
    global ok_all
    ok = got == want
    ok_all &= ok
    print(f"  {label:52s} expected {want:>8}   got {got:>11}   "
          f"{'PASS' if ok else 'FAIL'}")


# ======================================================================
# the sweep
# ======================================================================
def load_sweep(name="threshold_sweep.csv"):
    rows = list(csv.DictReader(open(os.path.join(DATA, name))))
    out = defaultdict(dict)
    for r in rows:
        tp, fp, fn = int(r["tp"]), int(r["fp"]), int(r["fn"])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        out[(r["variant"], r["recording"])][round(float(r["threshold"]), 3)] = \
            dict(prec=prec, rec=rec, f1=f1, n_ref=int(r["n_reference"]),
                 det=int(r["n_detected"]), tp=tp, fp=fp, fn=fn,
                 alpha=float(r["alpha"]))
    return len(rows), out


def mean_at(S, variant, th, field):
    return float(np.mean([S[(variant, s)][th][field] for s in SUBJECTS]))


def best_prec_at_recall(rows, target):
    """`compare.py.at_recall`: the best precision reachable at recall >= target.

    Recall is NOT monotone in the threshold - the 2 s refractory lets a weak
    early crossing block a strong later one - so this is the max over the
    feasible set, which is the standard interpolated precision and is monotone
    by construction.  `None` if the recall is never reached.
    """
    ok = [r for r in rows if r["rec"] >= target - 1e-9]
    return max(r["prec"] for r in ok) if ok else None


def mean_best_prec(S, variant, ths, target):
    """compare.py's rule: per recording take the max, THEN average."""
    v = [best_prec_at_recall([S[(variant, s)][t] for t in ths], target)
         for s in SUBJECTS]
    return None if any(x is None for x in v) else float(np.mean(v))


def check_sweep():
    n, S = load_sweep()
    print("\n=== Deliverable B: threshold_sweep.csv ===")
    mark_eq(n, 1251, "row count")
    ths = sorted(S[("frozen", "DH3")])
    mark_eq(len(ths), 139, "thresholds per (variant, recording)")
    mark_eq(f"{ths[0]:.3f}-{ths[-1]:.3f}", "0.300-0.990", "threshold range")

    # the published curves were computed on 0.30..0.98 step 0.01; that grid is a
    # subset of this one, so 5.4 is checked on it and reported on both
    coarse = [t for t in ths if int(round(t * 1000)) % 10 == 0 and t <= 0.98]

    print("\n--- 5.5 reference counts in the scored span ---")
    for s, want in zip(SUBJECTS, (261, 185, 495)):
        mark_eq(S[("frozen", s)][0.70]["n_ref"], want, f"n_reference {s}")

    print("\n--- 5.2 mean across recordings at threshold 0.70 ---")
    for variant, wp, wr, wf in (("frozen", 0.316, 0.366, 0.339),
                                ("correction_nrem_seeddrop", 0.388, 0.392, 0.383)):
        mark(mean_at(S, variant, 0.70, "prec"), wp, 5e-4, f"{variant} mean precision")
        mark(mean_at(S, variant, 0.70, "rec"), wr, 5e-4, f"{variant} mean recall")
        mark(mean_at(S, variant, 0.70, "f1"), wf, 5e-4, f"{variant} mean F1")
    mark(mean_at(S, "correction_crops", 0.70, "prec"), 0.382, 5e-4,
         "correction_crops mean precision")
    mark(mean_at(S, "correction_crops", 0.70, "f1"), 0.346, 5e-4,
         "correction_crops mean F1")

    print("\n--- 5.2 per recording, F1 at 0.70 ---")
    for s, wf, wn in zip(SUBJECTS, (0.334, 0.249, 0.434), (0.426, 0.268, 0.455)):
        mark(S[("frozen", s)][0.70]["f1"], wf, 5e-4, f"frozen {s} F1")
        mark(S[("correction_nrem_seeddrop", s)][0.70]["f1"], wn, 5e-4,
             f"correction_nrem_seeddrop {s} F1")

    print("\n--- 5.3 sweep shape, correction_nrem_seeddrop, mean ---")
    want = {0.66: (0.332, 0.403, 0.355), 0.68: (0.361, 0.402, 0.372),
            0.70: (0.388, 0.392, 0.383), 0.72: (0.388, 0.334, 0.353),
            0.74: (0.392, 0.280, 0.323), 0.76: (0.391, 0.216, 0.276),
            0.78: (0.393, 0.157, 0.223), 0.80: (0.385, 0.107, 0.164)}
    for th, (wp, wr, wf) in want.items():
        for field, w in (("prec", wp), ("rec", wr), ("f1", wf)):
            mark(mean_at(S, "correction_nrem_seeddrop", th, field), w, 5e-4,
                 f"th {th:.2f} mean {field}")

    print("\n--- 5.4 best precision reachable at recall >= target "
          "(compare.py's rule, on the published 0.01 subgrid) ---")
    want = {"frozen": (0.362, 0.333, None, 0.455),
            "correction_crops": (0.415, 0.384, None, 0.514),
            "correction_nrem_seeddrop": (0.424, 0.402, 0.424, 0.556)}
    for variant, (w25, w30, wdh3, wmf7) in want.items():
        mark(mean_best_prec(S, variant, coarse, 0.25), w25, 5e-4,
             f"{variant} mean recall>=0.25")
        mark(mean_best_prec(S, variant, coarse, 0.30), w30, 5e-4,
             f"{variant} mean recall>=0.30")
        for s, w in (("DH3", wdh3), ("MF7", wmf7)):
            got = best_prec_at_recall([S[(variant, s)][t] for t in coarse], 0.45)
            mark_reach(got, w, f"{variant} {s} recall>=0.45")

    print("\n--- 5.4 the same on the full 0.005 grid (finer grid, so >= the "
          "published value by construction; not a check) ---")
    for variant in want:
        for target in (0.25, 0.30):
            v = mean_best_prec(S, variant, ths, target)
            print(f"  {variant:40s} recall>={target:.2f}   "
                  + ("unreachable" if v is None else f"{v:.3f}"))
        for s in ("DH3", "MF7"):
            got = best_prec_at_recall([S[(variant, s)][t] for t in ths], 0.45)
            print(f"  {variant + ' ' + s:40s} recall>=0.45   "
                  + ("unreachable" if got is None else f"{got:.3f}"))

    print("\n--- 5.4 correction_crops as RESULTS.md 9 defines it: the "
          "`expandg10` lab correction, `joint` pick "
          "(threshold_sweep_published_pr.csv) ---")
    n2, S2 = load_sweep("threshold_sweep_published_pr.csv")
    mark_eq(n2, 417, "supplementary row count")
    v = "correction_crops_expandg10_joint"
    ths2 = sorted(S2[(v, "DH3")])
    coarse2 = [t for t in ths2 if int(round(t * 1000)) % 10 == 0 and t <= 0.98]
    mark(mean_best_prec(S2, v, coarse2, 0.25), 0.415, 5e-4, f"{v} mean recall>=0.25")
    mark(mean_best_prec(S2, v, coarse2, 0.30), 0.384, 5e-4, f"{v} mean recall>=0.30")
    mark_reach(best_prec_at_recall([S2[(v, "DH3")][t] for t in coarse2], 0.45),
               None, f"{v} DH3 recall>=0.45")
    mark_reach(best_prec_at_recall([S2[(v, "MF7")][t] for t in coarse2], 0.45),
               0.514, f"{v} MF7 recall>=0.45")
    return S


def mark_reach(got, want, label):
    """Like `mark`, but `want=None` means the recall must be unreachable."""
    global ok_all
    if want is None:
        ok = got is None
        ok_all &= ok
        print(f"  {label:52s} expected {'unreach':>8s}   got "
              f"{'unreachable' if got is None else f'{got:.3f}':>11s}   "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        mark(got, want, 5e-4, label)


# ======================================================================
# latency
# ======================================================================
COMPONENTS = (("t_filter_ms", "filter + detrend + normalize", 0.17, 0.22, 0.35),
              ("t_features_ms", "feature extraction", 10.17, 11.00, 12.8),
              ("t_model1_ms", "frozen classifier", 0.40, 0.54, 0.72),
              ("t_model2_ms", "correction", 0.14, 0.19, 0.32),
              ("t_compute_ms", "compute", 10.89, 11.86, 14.1),
              ("t_end_to_end_ms", "end-to-end", 35.9, 36.9, 39.1))


def check_latency():
    path = os.path.join(DATA, "latency_windows.csv")
    if not os.path.exists(path):
        print("\n=== Deliverable A: latency_windows.csv MISSING ===")
        return
    rows = list(csv.DictReader(open(path)))
    print("\n=== Deliverable A: latency_windows.csv ===")
    mark_eq(len(rows), 22671, "row count")
    per = {s: {c: np.array([float(r[c]) for r in rows if r["recording"] == s])
               for c, *_ in COMPONENTS} for s in SUBJECTS}
    pool = {c: np.array([float(r[c]) for r in rows]) for c, *_ in COMPONENTS}
    for s in SUBJECTS:
        print(f"  {s}: {len(per[s]['t_compute_ms'])} windows")

    print("\n--- 5.1 RESULTS.md convention: mean over the three per-recording "
          "statistics ---")
    for col, label, wmed, wp95, wp99 in COMPONENTS:
        for stat, want, f in (("median", wmed, np.median),
                              ("p95", wp95, lambda v: np.percentile(v, 95)),
                              ("p99", wp99, lambda v: np.percentile(v, 99))):
            got = float(np.mean([f(per[s][col]) for s in SUBJECTS]))
            mark(got, want, 0.055 if want < 1 else 0.06, f"{label} {stat}",
                 "{:.2f}")

    print("\n--- the same statistics over the pooled 22 671 windows (for "
          "reference; §5.1's table is not pooled) ---")
    print(f"  {'component':32s}{'median':>9s}{'p95':>9s}{'p99':>9s}{'max':>9s}")
    for col, label, *_ in COMPONENTS:
        v = pool[col]
        print(f"  {label:32s}{np.median(v):9.2f}{np.percentile(v, 95):9.2f}"
              f"{np.percentile(v, 99):9.2f}{v.max():9.2f}")

    print("\n--- the 250 ms budget ---")
    for col in ("t_compute_ms", "t_end_to_end_ms"):
        over = [r for r in rows if float(r[col]) > 250.0]
        print(f"  {col:20s} max {pool[col].max():8.2f} ms   over 250 ms: "
              f"{len(over)}  " + ", ".join(f"{r['recording']}#{r['window_index']}"
                                           f"={float(r[col]):.1f}" for r in over))
    fmax = pool["t_features_ms"].argmax()
    print(f"  feature-extraction max {pool['t_features_ms'].max():.2f} ms on "
          f"{rows[fmax]['recording']} window {rows[fmax]['window_index']}")


if __name__ == "__main__":
    check_sweep()
    check_latency()
    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
