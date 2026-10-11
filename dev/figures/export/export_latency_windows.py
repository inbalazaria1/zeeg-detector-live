#!/usr/bin/env python
"""
Deliverable A of `DATA_EXPORT_BRIEF.md`: per-window latency.

`sleep/latency.py` already times every window - `sim.t_filter`, `sim.t_feat`,
`sim.t_model`, `sim.t_model2`, `sim.t_total` are one entry per 250 ms decision -
but it only ever wrote the summary statistics out.  This re-runs it, unchanged,
through its own `shipped_corrector` / `feed_for` / `TwoModel` / `report`, and
saves the arrays.  Nothing about the chain, the models or the feed is redefined
here; if this file and `latency.py` ever disagree, `latency.py` is right.

    RESEARCH_PACKAGE=/path/to/research-package ./.venv/bin/python export_latency_windows.py

Run it on an otherwise idle machine.  The medians are stable, the p99 and the
max are not - they are whatever else the laptop was doing.
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
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "latency_windows.csv"))

sys.path.insert(0, SLEEP)
import latency as L                                             # noqa: E402
import realtime_fir101 as RT                                    # noqa: E402
import windows as W                                             # noqa: E402

# the batch detection counts every table in RESULTS.md is computed from; the
# streamed chain reproduced them exactly (RESULTS.md 14, "cross-check")
BATCH_FIRED = {"DH3": 359, "AO0": 196, "MF7": 366}


def main():
    import json
    with open(os.path.join(SLEEP, "det.json")) as fh:
        det = json.load(fh)["res"]["nrem_ns"]

    lines = []

    def log(*s):
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        lines.append(m)

    log(f"stage 1: {RT.FIR_TAPS}-tap firwin ({RT.LOOKAHEAD_MS:.0f} ms) behind a "
        f"causal order-{L.HP_ORDER} 0.1 Hz Butterworth")

    rows = []
    for s in W.SUBJECTS:
        log(f"\n=== {s} ===")
        picks = det[s]["hold70"]
        corr = L.shipped_corrector(s, picks, log)
        feed, warm, n_crop = L.feed_for(s, 1200.0)
        sim = L.TwoModel(feed, warm, n_crop, corr=corr, alpha=picks["alpha"])
        sim.fit_normalisation(verbose=True)
        r = sim.run(verbose=True)
        L.report(s, sim, r, log)

        n = len(sim.t_total)
        log(f"  detections streamed: {len(r['fired'])}   batch: "
            f"{BATCH_FIRED[s]}   "
            + ("MATCH" if len(r["fired"]) == BATCH_FIRED[s] else "DIFFER"))
        for i in range(n):
            tot = sim.t_total[i]
            rows.append((s, i, sim.t_filter[i], sim.t_feat[i], sim.t_model[i],
                         sim.t_model2[i], tot, tot + RT.LOOKAHEAD_MS))
        del sim, feed, corr

    cols = ["recording", "window_index", "t_filter_ms", "t_features_ms",
            "t_model1_ms", "t_model2_ms", "t_compute_ms", "t_end_to_end_ms"]
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[0], r[1]] + [repr(float(v)) for v in r[2:]])
    log(f"\n{len(rows)} rows -> {OUT}")

    with open(OUT.replace(".csv", ".log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
