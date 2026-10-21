#!/usr/bin/env python
"""
The whole night's E227 / E254 at 1000 Hz, and a window of it in the form
`pipeline.configure` wants.

The night comes from `_ext/cycles/cache/` if it is already there, otherwise it
is read out of the `.mff` once by `cache_night.fetch` into `cache/night/`.
`window_npz` then cuts one window, with 1 min of margin each side, and caches
it too.

From dev/research, to read a whole night ahead of time:

    ../../.venv/bin/python sleep/nightraw.py --fetch XY1
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRIAL = os.path.dirname(HERE)
ROOT = os.path.join(TRIAL, "_ext")
for _p in (os.path.join(ROOT, "cycles"), os.path.join(ROOT, "mf7")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = os.path.join(TRIAL, "cache", "night")
CHANS = ["E227", "E254"]
MARGIN_MIN = 1.0                 # read margin each side of a window

# the recording behind each subject; AO0 is the EA3001 .mff
REC = {"DH3": "DH3", "AO0": "EA3001", "MF7": "MF7", "ON68": "ON68"}

# whole nights already cached, read only: (file, start, end) in minutes
CYCLES_CACHE = os.path.join(ROOT, "cycles", "cache")
EXISTING = {
    "DH3": ("DH3_night_0_274.6.npz", 0.0, 274.6),
    "AO0": ("EA3001_night_0_398.933.npz", 0.0, 398.933),
    "MF7": ("MF7_night_0_173.npz", 0.0, 173.0),
    # its cached night does not start at minute 0, so it is always re-read
    "ON68": ("__none__", 0.0, -1.0),
}

# how far into each recording it is safe to read.  DH3's file has a gap at
# minute ~275 that the block arithmetic in mff_direct cannot cross.
MAX_MIN = {"DH3": 274.6, "AO0": 398.933, "MF7": 366.2, "ON68": 397.8}

# a subject listed only in hypno.SUBJECTS takes its recording and cap from
# there, and is always read from its .mff
import hypno as H                                                # noqa: E402
for _s, _cfg in H.SUBJECTS.items():
    REC.setdefault(_s, _cfg["rec"])
    MAX_MIN.setdefault(_s, _cfg["max_min"])
    EXISTING.setdefault(_s, ("__none__", 0.0, -1.0))

_NIGHT = {}


def night_path(subject, need_min):
    """A cached night covering minute 0 to `need_min`, read from the .mff if needed."""
    name, lo, hi = EXISTING[subject]
    p = os.path.join(CYCLES_CACHE, name)
    if os.path.exists(p) and hi >= need_min - 1e-6:
        return p
    mine = [f for f in os.listdir(CACHE) if f.startswith(f"{REC[subject]}_night_")
            ] if os.path.isdir(CACHE) else []
    for f in mine:
        end = float(f[:-4].split("_")[-1])
        if end >= need_min - 1e-6:
            return os.path.join(CACHE, f)
    import cache_night as CN
    CN.CACHE = CACHE
    return CN.fetch(subject, 0.0, MAX_MIN[subject], workers=16)


def night(subject, need_min=None):
    """The whole night as one array, kept in memory (250-400 MB each)."""
    need = MAX_MIN[subject] if need_min is None else need_min
    key = REC[subject]
    if key not in _NIGHT or _NIGHT[key][1] < need - 1e-6:
        p = night_path(subject, need)
        z = np.load(p, allow_pickle=True)
        assert list(z["ch_names"]) == CHANS and float(z["sfreq"]) == 1000.0
        _NIGHT[key] = (z["data"], float(z["crop_min"][1]))
    return _NIGHT[key][0]


def window_npz(subject, lo_min, hi_min, start_min=None):
    """`(npz path, cache_start_min)` for one window, with 1 min of margin.

    `start_min` moves where the cached block starts.  Only needed to reproduce
    an old matrix exactly, since the filter ring-up depends on the start.
    """
    os.makedirs(CACHE, exist_ok=True)
    a = max(0.0, lo_min - MARGIN_MIN) if start_min is None else float(start_min)
    b = min(MAX_MIN[subject], hi_min + MARGIN_MIN)
    out = os.path.join(CACHE, f"{REC[subject]}_w_{a:g}_{b:g}.npz")
    if os.path.exists(out):
        return out, a
    d = night(subject, b)
    i0, i1 = int(round(a * 60_000)), int(round(b * 60_000))
    seg = np.ascontiguousarray(d[:, i0:i1 + 1])
    tmp = os.path.join(CACHE, f".tmp-{os.getpid()}.npz")
    np.savez(tmp, data=seg, ch_names=np.array(CHANS), sfreq=1000.0,
             crop_min=np.array([a, b]))
    os.replace(tmp, out)
    return out, a


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", nargs="*", default=[],
                    help="recordings whose whole night to make sure exists")
    a = ap.parse_args()
    for s in a.fetch:
        print(s, "->", night_path(s, MAX_MIN[s]), flush=True)
