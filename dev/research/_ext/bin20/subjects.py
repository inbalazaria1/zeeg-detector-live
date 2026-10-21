#!/usr/bin/env python
"""
The three patients' lab crops behind one `configure()`.

    AO0   4    - 47.5 min
    DH3  23    - 74   min
    MF7   2    - 62   min

The reference for each is the original notebook's chain over its crop.  No
recording has expert spike labels, so scores measure what causality costs,
not how good the detector is.

`configure` and `reference_meta` need the lab-crop caches, which are not
included here; the training build does not call them.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "simulator-without-cleaning", "rt")
sys.path.insert(0, RT)

import pipeline as P                                            # noqa: E402

# (tmin_min, tmax_min), as the lab gave them
WINDOW = {"AO0": (4.0, 47.5), "DH3": (23.0, 74.0), "MF7": (2.0, 62.0)}

# calibration = one 20 min detection bin
CALIB_MIN = 20.0
CALIB_SEC = CALIB_MIN * 60.0


def configure(name):
    """Point `pipeline` at `name`'s lab crop; return the reference times."""
    tmin, tmax = WINDOW[name]
    if name == "DH3":
        P.configure(os.path.join(RT, "cache", "DH3_raw_22_75.npz"), 22.0,
                    tmin, tmax, None)
    elif name == "AO0":
        sys.path.insert(0, os.path.join(ROOT, "ao0"))
        import ao0_common as A
        A.configure(tmin, tmax)
    elif name == "MF7":
        sys.path.insert(0, os.path.join(ROOT, "mf7"))
        import mf7_common as M
        M.configure(tmin, tmax)
    else:
        raise KeyError(name)
    return P.load_baseline_times()


def reference_meta(name):
    """Spikes per minute per bin of the offline reference, if cached."""
    tmin, tmax = WINDOW[name]
    path = {"AO0": os.path.join(ROOT, "ao0", "cache",
                                f"AO0_offline_{tmin:g}_{tmax:g}.json"),
            "MF7": os.path.join(ROOT, "mf7", "cache",
                                f"MF7_offline_{tmin:g}_{tmax:g}.json")}.get(name)
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None
