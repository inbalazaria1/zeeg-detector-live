#!/usr/bin/env python
"""
Cached feature matrices for the lab crops, from `chains25` and `norms`.

The build in `sleep/build.py` does not use this; `build_x` needs research-tree
caches that are not included here. Cache files are
`cache/<SUBJ>/<chain>__<norm>[__cal<M>].pkl`, with the canonical chain name, so
`b25_o3` and `o3_t51` share one file.

Building runs in a fresh subprocess: `pipeline.epochs_to_features` forks a
pool, and forking after XGBoost is loaded deadlocks.
"""
import argparse
import os
import subprocess
import sys
import time

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
import chains25 as CH                                            # noqa: E402
import norms as NM                                               # noqa: E402

CACHE = os.path.join(HERE, "cache")

# research-tree caches, read only, searched in this order (not included here)
FOREIGN_CACHES = (os.path.join(ROOT, "filter_variants_50ms", "cache"),
                  os.path.join(ROOT, "bin20", "cache"))

# the same chains under the names those caches used; shown identical on real
# data in the research tree
FOREIGN_NAMES = {
    "o3_t51": ("i25_butter3",),
    "o2_t51": ("c25_fir51",),
}


# ======================================================================
# keys
# ======================================================================
def calib_suffix():
    return "" if abs(S.CALIB_MIN - 20.0) < 1e-9 else f"__cal{S.CALIB_MIN:g}"


def cache_tag(chain_name, norm_spec):
    return (f"{CH.parse(chain_name).name()}"
            f"__{NM.parse(norm_spec).name()}{calib_suffix()}")


def cache_path(subject, chain_name, norm_spec):
    return os.path.join(CACHE, subject, f"{cache_tag(chain_name, norm_spec)}.pkl")


def foreign_path(subject, chain_name, norm_spec):
    """The research-tree cache file for this cell, or None."""
    canon = CH.parse(chain_name).name()
    norm = NM.parse(norm_spec).name()
    names = (canon,) + FOREIGN_NAMES.get(canon, ())
    for root in FOREIGN_CACHES:
        for nm in names:
            p = os.path.join(root, subject, f"{nm}__{norm}{calib_suffix()}.pkl")
            if os.path.exists(p):
                return p
    return None


# ======================================================================
# building
# ======================================================================
def _write_atomic(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    x.to_pickle(tmp)
    os.replace(tmp, path)


def build_in_process(subject, chain_name, norm_spec, quiet=False):
    """Build the matrix in this process. Forks a pool, so run it in a child."""
    bp, detrend, _, _ = CH.get(chain_name)
    spec = NM.parse(norm_spec)
    t0 = time.time()
    filt = P.crop(bp(P.load_raw())[0])
    det = detrend(filt)[0]
    feats = {}
    for ch_i, ch in enumerate(P.CHANS):
        eps, chan = NM.build_epochs(det[ch_i], spec)
        df = P.epochs_to_features(eps, f"{subject}_Crop")
        df["chan_name"] = ch
        for k, v in chan.items():
            df[k] = v
        feats[ch] = df
    e1, e2 = feats[P.CHANS[0]], feats[P.CHANS[1]]
    x = pd.concat([e1, e2], axis=1, ignore_index=True)
    x.columns = ([f"eog1_{c}" for c in e1.columns]
                 + [f"eog2_{c}" for c in e2.columns])
    # load the model only after the last fork
    x = x[P.model().get_booster().feature_names]
    if not quiet:
        print(f"    built {subject}/{cache_tag(chain_name, norm_spec)}: "
              f"{x.shape} in {time.time() - t0:.0f}s", flush=True)
    return x


def build_x(subject, chain_name, norm_spec, quiet=False):
    """The matrix for one lab crop. Needs research-tree caches not included here.

    Tries this cache, then the research-tree caches, then builds it in a
    subprocess.
    """
    path = cache_path(subject, chain_name, norm_spec)
    if os.path.exists(path):
        return pd.read_pickle(path)
    ro = foreign_path(subject, chain_name, norm_spec)
    if ro is not None:
        if not quiet:
            print(f"    reusing {os.path.relpath(ro, ROOT)}", flush=True)
        return pd.read_pickle(ro)

    tag = cache_tag(chain_name, norm_spec)
    if not quiet:
        print(f"    building {subject}/{tag} in a subprocess ...", flush=True)
    cmd = [sys.executable, os.path.join(HERE, "feats.py"), "--build",
           "--subject", subject, "--chain", chain_name,
           "--norm", NM.parse(norm_spec).name(),
           "--calib-min", f"{S.CALIB_MIN:g}"]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"feature build failed for {subject}/{tag} "
                           f"(exit {r.returncode})")
    return pd.read_pickle(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--calib-min", type=float, default=20.0)
    a = ap.parse_args()

    S.CALIB_MIN = float(a.calib_min)
    S.CALIB_SEC = S.CALIB_MIN * 60.0
    S.configure(a.subject)
    x = build_in_process(a.subject, a.chain, a.norm)
    _write_atomic(x, cache_path(a.subject, a.chain, a.norm))


if __name__ == "__main__":
    main()
