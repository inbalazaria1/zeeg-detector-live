#!/usr/bin/env python
"""
The training matrices for any [lo, hi] minutes of a night, cached in
`cache/sleep/`.

Two kinds per window:

  causal    the chain the detector runs (25 ms lookahead, expanding z-score
            seeded on the window's first 20 min).  Model 1 on this is the
            frozen score.
  offline   the original notebook's chain (16.5 s lookahead, whole-window
            z-score).  Model 1 on this is the teacher, and `pipeline.detect`
            of it at 0.70 is the reference.

Each window is built in its own subprocess: feature extraction forks a pool,
and forking after XGBoost is loaded deadlocks, so the model is loaded last and
the child exits.

From dev/research:

    ../../.venv/bin/python sleep/build.py --plan
    ../../.venv/bin/python sleep/build.py --all
    ../../.venv/bin/python sleep/build.py --one DH3 c2 74.8 116.983 causal
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRIAL = os.path.dirname(HERE)
ROOT = os.path.join(TRIAL, "_ext")
for _p in (HERE, TRIAL, os.path.join(TRIAL, "norm"),
           os.path.join(ROOT, "bin20"),
           os.path.join(ROOT, "simulator-without-cleaning", "rt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as P                                             # noqa: E402
import chains25 as CH                                            # noqa: E402
import norms as NM                                               # noqa: E402
import norms6 as NM6                                             # noqa: E402
import windows as W                                              # noqa: E402
import nightraw as NR                                            # noqa: E402

CACHE = os.path.join(TRIAL, "cache", "sleep")
CHAIN = "b25_o3"

# `causal` is the shipped chain.  `causalg` is the same chain with the z-score
# seeded on 10 s instead of 20 min: it changes only the seed block, never a
# scored epoch.  It was tried and not shipped; train with `causal`.
NORM = {"causal": "expand20", "causalg": "expandg10"}
NORM_MOD = {"causal": NM, "causalg": NM6}
KINDS = ("causal", "causalg", "offline")


# ======================================================================
# paths
# ======================================================================
def tag_of(subject, tag, lo, hi, kind):
    return f"{tag}_{lo:g}_{hi:g}__{kind}"


def path_of(subject, tag, lo, hi, kind):
    return os.path.join(CACHE, subject, tag_of(subject, tag, lo, hi, kind) + ".pkl")


def _write_atomic(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    x.to_pickle(tmp)
    os.replace(tmp, path)


# ======================================================================
# the build, run in a child: forks first, loads the model last
# ======================================================================
def configure(subject, lo, hi, start_min=None):
    """Point `pipeline` at [lo, hi] min of `subject`, out of the cached night."""
    npz, start = NR.window_npz(subject, lo, hi, start_min)
    P.configure(npz, start, lo, hi, None)
    return npz


def build_in_process(subject, lo, hi, kind, quiet=False):
    t0 = time.time()
    raw = P.load_raw()
    if kind in NORM:
        bp, detrend, _, _ = CH.get(CHAIN)
        filt = P.crop(bp(raw)[0])
        det = detrend(filt)[0]
        mod = NORM_MOD[kind]
        spec = mod.parse(NORM[kind])
        epochs = [mod.build_epochs(det[i], spec) for i in range(len(P.CHANS))]
    elif kind == "offline":
        filt = P.crop(P.BandPass.mne_zero(raw)[0])
        det = P.detrend_sma(filt, look_ms=None)[0]
        norm = P.Normalizer("global")
        epochs = [P.build_epochs(det[i], norm) for i in range(len(P.CHANS))]
    else:
        raise KeyError(kind)

    feats = {}
    for ch_i, ch in enumerate(P.CHANS):
        eps, chan = epochs[ch_i]
        df = P.epochs_to_features(eps, f"{subject}_Crop")
        df["chan_name"] = ch
        for k, v in chan.items():
            df[k] = v
        feats[ch] = df
    e1, e2 = feats[P.CHANS[0]], feats[P.CHANS[1]]
    x = pd.concat([e1, e2], axis=1, ignore_index=True)
    x.columns = ([f"eog1_{c}" for c in e1.columns]
                 + [f"eog2_{c}" for c in e2.columns])
    # model loaded only now, after the last fork
    x = x[P.model().get_booster().feature_names]
    if not quiet:
        print(f"    {subject} {lo:g}-{hi:g} {kind}: {x.shape} "
              f"in {time.time() - t0:.0f}s", flush=True)
    return x


# ======================================================================
# the driver - one child per (window, kind)
# ======================================================================
def get(subject, tag, lo, hi, kind, build=True):
    """One window's matrix, built in a subprocess if it is not cached."""
    p = path_of(subject, tag, lo, hi, kind)
    if os.path.exists(p):
        return pd.read_pickle(p)
    if not build:
        raise FileNotFoundError(p)
    cmd = [sys.executable, os.path.abspath(__file__), "--one",
           subject, tag, f"{lo:g}", f"{hi:g}", kind]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(p):
        raise RuntimeError(f"build failed: {subject} {tag} {kind} "
                           f"(exit {r.returncode})")
    return pd.read_pickle(p)


def plan(kinds=KINDS, subjects=None):
    """Every (subject, tag, lo, hi, kind) to build, and whether it is cached."""
    out = []
    for s in (subjects or W.ALL):
        for tag, lo, hi, dur in W.long_windows(s):
            for kind in kinds:
                out.append((s, tag, lo, hi, dur, kind,
                            os.path.exists(path_of(s, tag, lo, hi, kind))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=5, metavar=("SUBJ", "TAG", "LO", "HI", "KIND"))
    ap.add_argument("--start-min", type=float, default=None,
                    help="override the cached block's start (self-test only)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--subjects", default=",".join(W.ALL))
    a = ap.parse_args()

    if a.one:
        s, tag, lo, hi, kind = a.one
        lo, hi = float(lo), float(hi)
        configure(s, lo, hi, a.start_min)
        x = build_in_process(s, lo, hi, kind)
        _write_atomic(x, path_of(s, tag, lo, hi, kind))
        return

    rows = plan([k for k in a.kinds.split(",") if k],
                [s for s in a.subjects.split(",") if s])
    todo = [r for r in rows if not r[6]]
    mins = sum(r[4] for r in todo)
    print(f"{len(rows)} cells, {len(todo)} missing ({mins:.0f} window-min of "
          f"feature extraction)", flush=True)
    for s, tag, lo, hi, dur, kind, have in rows:
        print(f"  {s:4s} {tag:>3s} {lo:6.1f}-{hi:6.1f} {dur:5.1f} min "
              f"{kind:8s} {'ok' if have else 'MISSING'}")
    if not a.all:
        return

    t0 = time.time()
    for i, (s, tag, lo, hi, dur, kind, have) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {s} {tag} {lo:g}-{hi:g} {kind} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        get(s, tag, lo, hi, kind)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
