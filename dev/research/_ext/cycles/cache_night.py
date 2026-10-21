#!/usr/bin/env python
"""
Read E227 / E254 of a whole night out of the .mff, in parallel, into one .npz.

One read per block covers both channels, and a thread pool runs the blocks
side by side.  Both are there because the recordings sat on a Google Drive
mount, where the cost is per request, not per byte.  The run can be stopped
and resumed: progress is kept in a `.part.npy` plus a `.part.done.npy` mask.

Block layout and scaling come from `mff_direct.py`, which matches MNE.  The
.npz holds `data`, `ch_names`, `sfreq` and `crop_min`.

From dev/research (sleep/nightraw.py calls this for you):

    ../../.venv/bin/python _ext/cycles/cache_night.py XY1
    ../../.venv/bin/python _ext/cycles/cache_night.py XY1 --workers 24
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mf7"))
sys.path.insert(0, HERE)

from mff_direct import MffSignal, channel_index, read_gcal              # noqa: E402
import hypno as H                                                       # noqa: E402

CACHE = os.path.join(HERE, "cache")
CHANS = ["E227", "E254"]
TRIES = 40                    # retries per block; a Drive mount drops reads often
BACKOFF = 2.0


def rec_key(name):
    """Cache files are named by recording, so AO0 and EA3001 share one."""
    return H.SUBJECTS[name]["rec"]


def npz_path(name, start_min, end_min):
    return os.path.join(CACHE,
                        f"{rec_key(name)}_night_{start_min:g}_{end_min:g}.npz")


def fetch(name, start_min=0.0, end_min=None, workers=16, quiet=False):
    """Cache `name`'s two channels over [start_min, end_min); return the .npz path."""
    mff = os.path.join(ROOT, H.SUBJECTS[name]["mff"])
    os.makedirs(CACHE, exist_ok=True)

    sig = MffSignal(os.path.join(mff, "signal1.bin"))
    if end_min is None:
        end_min = sig.n_times / sig.sfreq / 60.0
    out = npz_path(name, start_min, end_min)
    if os.path.exists(out):
        if not quiet:
            print(f"{name}: already cached -> {out}")
        return out
    part, donef = out[:-4] + ".part.npy", out[:-4] + ".part.done.npy"

    gcal = read_gcal(os.path.join(mff, "info1.xml"))
    idx = channel_index(mff, CHANS)
    s = sig.samples_per_block
    per = s * 4                                   # bytes of one channel per block
    off = [int(sig.offsets[i]) for i in idx]
    lo_off, hi_off = min(off), max(off) + per     # one read covers both channels
    rel = [o - lo_off for o in off]
    want = hi_off - lo_off

    start = int(round(start_min * 60 * sig.sfreq))
    stop = min(int(round(end_min * 60 * sig.sfreq)), sig.n_times)
    k0, k1 = start // s, (stop - 1) // s + 1
    n = stop - start

    if os.path.exists(part) and os.path.exists(donef):
        buf = np.lib.format.open_memmap(part, mode="r+")
        done = np.load(donef)
        assert buf.shape == (len(idx), n) and len(done) == k1 - k0, \
            f"{part} does not match this request; delete it and re-run"
    else:
        buf = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32,
                                        shape=(len(idx), n))
        done = np.zeros(k1 - k0, dtype=bool)

    todo = list(np.flatnonzero(~done) + k0)
    if not quiet:
        print(f"{name}: min {start_min:g}-{end_min:g}, blocks {k0}-{k1 - 1}, "
              f"{len(todo)} of {k1 - k0} missing, one {want / 1e3:.0f} kB read "
              f"each ({len(todo) * want / 1e9:.1f} GB), {workers} workers",
              flush=True)
    if not todo:
        return _finish(out, part, donef, buf, done, gcal, idx, sig,
                       start_min, end_min, quiet)

    # block offsets are computed, not read, so spot-check them first
    for attempt in range(TRIES):
        try:
            sig.check_markers(sorted({k0, k1 - 1, *range(k0, k1, 256)}))
            break
        except OSError:
            if attempt == TRIES - 1:
                raise
            time.sleep(BACKOFF)

    path = os.path.join(mff, "signal1.bin")
    lock = threading.Lock()
    local = threading.local()
    state = {"n": 0, "t0": time.time()}

    def handle(k):
        fh = getattr(local, "fh", None)
        if fh is None:
            fh = local.fh = open(path, "rb")     # one file handle per thread,
        base = sig.data_start(k)                 # so seeks don't collide
        b0 = k * s
        a, b = max(start, b0) - b0, min(stop, b0 + s) - b0
        for attempt in range(TRIES):
            try:
                fh.seek(base + lo_off)
                raw = fh.read(want)
                if len(raw) == want:
                    break
                err = f"short read {len(raw)}/{want}"
            except OSError as e:
                err = repr(e)
            if attempt == TRIES - 1:
                raise IOError(f"block {k}: {err} after {TRIES} tries")
            time.sleep(BACKOFF)
        for j, r in enumerate(rel):
            ch = np.frombuffer(raw, dtype="<f4", count=s, offset=r)
            buf[j, b0 + a - start: b0 + b - start] = ch[a:b]
        with lock:
            done[k - k0] = True
            state["n"] += 1
            i = state["n"]
            if i % 100 == 0 or i == len(todo):
                buf.flush()
                np.save(donef, done)
                el = time.time() - state["t0"]
                if not quiet:
                    print(f"  {name} {i}/{len(todo)} blocks  {el:6.0f}s  "
                          f"{i * want / el / 1e6:5.2f} MB/s  "
                          f"eta {el / i * (len(todo) - i) / 60:6.1f} min",
                          flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(handle, todo))

    return _finish(out, part, donef, buf, done, gcal, idx, sig,
                   start_min, end_min, quiet)


def _finish(out, part, donef, buf, done, gcal, idx, sig, start_min, end_min, quiet):
    buf.flush()
    np.save(donef, done)
    assert done.all(), f"{(~done).sum()} blocks still missing"
    cal = np.array([1e-6 * (gcal[c] if gcal is not None else 1.0) for c in idx])
    data = np.asarray(buf, dtype=np.float64) * cal[:, None]
    np.savez(out, data=data, ch_names=np.array(CHANS), sfreq=sig.sfreq,
             crop_min=np.array([start_min, end_min]))
    del buf
    os.remove(part)
    os.remove(donef)
    if not quiet:
        print(f"{os.path.basename(out)}: saved {data.shape} @ {sig.sfreq} Hz "
              f"({os.path.getsize(out) / 1e9:.1f} GB)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cache a whole night, 2 channels")
    ap.add_argument("subject", nargs="?", choices=list(H.SUBJECTS))
    ap.add_argument("--all", action="store_true", help="every recording, in turn")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    names = list(H.SUBJECTS) if a.all else [a.subject]
    if not names or names == [None]:
        ap.error("give a subject or --all")
    for nm in names:
        fetch(nm, a.start, a.end, a.workers)
