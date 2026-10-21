#!/usr/bin/env python
"""
The offline notebook's pipeline (`spm_extraction_annotated.ipynb`), in memory.

The features are the notebook's own cell 15, executed as-is, and detection is a
line-for-line copy of its stage 7. Only the .mff -> .fif -> .pkl file shuffling
is gone, which saves ~3 min per run. `--self-test` reproduced the notebook's 409
DH3 spikes at identical times (needs the DH3 lab-crop cache, not included).

Stages, and how much future each one reads:

  1 band-pass 0.1-70 Hz     33001-tap zero-phase FIR      16.5 s
  2 crop                    -                             -
  3 SMA detrend             centred 2 s box               ~1 s
  6 z-score + chan_* stats  whole crop                    the whole crop
  6 features per 250 ms epoch                             -
  7 XGB, threshold 0.7, 2 s refractory, 20 min bins       -

`BandPass`, `detrend_sma` and `Normalizer` are the pieces a causal chain swaps.

    python pipeline.py --self-test
"""
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import scipy.signal
import scipy.signal as sp_signal
import mne.filter

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SIMDIR)
# cell 15 of the notebook, the feature code; the rest of it is Colab setup
NB_CELL_PATH = os.path.join(SIMDIR, "spm_features_cell15.py")
# model 1, byte-identical to the research tree's xgb_s50_f210_b_3000max.pkl
REPO_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir, os.pardir, os.pardir))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "xgb_frozen.pkl")
RAW_CACHE = os.path.join(HERE, "cache", "DH3_raw_22_75.npz")

# notebook constants, by notebook stage
SFREQ = 1000.0
CACHE_START_MIN = 22.0            # where the raw cache starts
TMIN_MIN, TMAX_MIN = 23, 74       # stage 2
BP_LOW, BP_HIGH = 0.1, 70.0       # stage 1
SMA_SEC = 2.0                     # stage 3
EPOCH = 250                       # stage 6, samples
CHANS = ["E227", "E254"]          # stage 8
THRESHOLD = 0.7                   # stage 9
SEC_LIMIT = 2                     # stage 9, refractory s
BIN_LENGTH = 20                   # stage 9, min
SAMPLES_PER_MIN = 240             # stage 7: epochs per minute
EPOCH_SEC = 0.25

# the notebook's spike count on DH3 23-74 min
BASELINE_N_SPIKES = 409

# reference spikes: None = DH3's, from compare_notebook_vs_sim (not included);
# configure() points it at a json for other subjects
BASELINE_JSON = None


def configure(raw_cache, cache_start_min, tmin_min, tmax_min, baseline_json=None):
    """Point the module at another recording and crop. Nothing else here is
    per-subject (`subj` is only a label column)."""
    global RAW_CACHE, CACHE_START_MIN, TMIN_MIN, TMAX_MIN, BASELINE_JSON
    RAW_CACHE = raw_cache
    CACHE_START_MIN = cache_start_min
    TMIN_MIN, TMAX_MIN = tmin_min, tmax_min
    BASELINE_JSON = baseline_json


# ===================================================================
# the notebook's feature code
# ===================================================================
_NB_NS = None


def nb_namespace():
    """Exec cell 15 once and return its namespace (np, pd, sp_stats injected)."""
    global _NB_NS
    if _NB_NS is None:
        with open(NB_CELL_PATH) as fh:
            cell = fh.read()
        ns = {"np": np, "pd": pd, "sp_stats": sp_stats}
        exec(compile(cell, "<nb cell 15>", "exec"), ns)
        assert "extract_epochs_features_mne" in ns
        _NB_NS = ns
    return _NB_NS


_POOL_EPOCHS = None


def _feat_chunk(arg):
    lo, hi = arg
    fn = nb_namespace()["extract_epochs_features_mne"]
    return fn(_POOL_EPOCHS[lo:hi], "s", SFREQ)


def _pool_init(epochs):
    global _POOL_EPOCHS
    _POOL_EPOCHS = epochs


def epochs_to_features(epochs, subj, n_jobs=None):
    """Cell 15's features for every epoch, split across processes.

    Every feature is per-epoch, so splitting is exact. Uses fork: load the
    XGBoost model only after this, or the fork deadlocks.
    """
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    n = len(epochs)
    if n_jobs == 1 or n < 4 * n_jobs:
        df = nb_namespace()["extract_epochs_features_mne"](epochs, subj, SFREQ)
    else:
        edges = np.linspace(0, n, n_jobs * 2 + 1).astype(int)
        jobs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ctx = mp.get_context("fork")
        with ctx.Pool(len(jobs), initializer=_pool_init, initargs=(epochs,)) as pool:
            parts = pool.map(_feat_chunk, jobs)
        df = pd.concat(parts, ignore_index=True)
    df["subj"] = subj
    df["epoch_id"] = np.arange(len(df))
    return df


# ===================================================================
# stage 1 - band-pass, each returns (filtered, lookahead_ms)
# ===================================================================
def _mne_kernel():
    """The 33001-tap kernel `raw.filter(0.1, 70)` builds at 1000 Hz."""
    return mne.filter.create_filter(
        None, SFREQ, BP_LOW, BP_HIGH, method="fir", fir_design="firwin",
        phase="zero", fir_window="hamming", verbose=False)


class BandPass:
    """Stage 1 variants, each `f(data) -> (out, lookahead_ms)`."""

    @staticmethod
    def mne_zero(data):
        """The notebook's: MNE zero-phase FIR, 16.5 s of future."""
        out = mne.filter.filter_data(data, SFREQ, BP_LOW, BP_HIGH, verbose=False)
        return out, 16500.0

    @staticmethod
    def firwin(data, ntaps=51, hp_iir=True):
        """Short symmetric low-pass applied centred, plus an optional causal
        0.1 Hz high-pass. Lookahead is (ntaps - 1) / 2 samples."""
        h = sp_signal.firwin(ntaps, BP_HIGH, fs=SFREQ)
        sos = sp_signal.butter(2, BP_LOW, btype="highpass", output="sos", fs=SFREQ)
        out = []
        for ch in data:
            ch = ch - np.nanmean(ch)
            if hp_iir:
                ch = sp_signal.sosfilt(sos, ch)
            out.append(np.convolve(ch, h, mode="same"))
        return np.array(out), (ntaps - 1) // 2 / SFREQ * 1000

    @staticmethod
    def trunc(data, look_ms=50.0, past_ms=None, renorm=True):
        """The notebook's kernel with its future taps cut at `look_ms`.

        Rescaled so the 1-40 Hz gain stays 1: cutting the tail loses a little
        gain, and every feature that is not scale-free would shift with it.
        """
        h = _mne_kernel()
        c = (len(h) - 1) // 2
        look = int(round(look_ms / 1000 * SFREQ))
        past = c if past_ms is None else min(c, int(round(past_ms / 1000 * SFREQ)))
        hh = h[c - look: c + past + 1].copy()          # future taps, then past taps
        if renorm:
            w, hf = sp_signal.freqz(hh, worN=4096, fs=SFREQ)
            band = (w >= 1) & (w <= 40)
            hh /= np.abs(hf[band]).mean()
        # y[n] = sum_j hh[j] x[n + look - j]: full convolution read at n + look
        out = np.array([
            sp_signal.fftconvolve(ch - np.nanmean(ch), hh, mode="full")[look:look + len(ch)]
            for ch in data])
        return out, look_ms

    @staticmethod
    def minphase(data, look_ms=50.0):
        """Minimum-phase version of the notebook's kernel, read `look_ms` late
        to pay back some of its group delay."""
        h = _mne_kernel()
        hmin = sp_signal.minimum_phase(h, method="hilbert")
        adv = int(round(look_ms / 1000 * SFREQ))
        out = []
        for ch in data:
            y = sp_signal.lfilter(hmin, 1.0, ch - np.nanmean(ch))
            out.append(np.concatenate([y[adv:], np.zeros(adv)]) if adv else y)
        return np.array(out), look_ms


# ===================================================================
# stage 3 - SMA detrend
# ===================================================================
def detrend_sma(data, look_ms=None, win_sec=SMA_SEC):
    """Subtract a `win_sec` moving average.

    look_ms=None is the notebook's centred window (~1 s of future); a number
    shifts the window so only that many ms sit ahead of the sample.
    """
    w = int(SFREQ * win_sec)
    k = np.ones(w) / w
    if look_ms is None:
        trend = np.array([np.convolve(ch, k, mode="same") for ch in data])
        return data - trend, (w // 2 - 1) / SFREQ * 1000
    nf = int(round(look_ms / 1000 * SFREQ))
    trend = np.array([np.convolve(ch, k, mode="full")[nf:nf + len(ch)] for ch in data])
    return data - trend, float(look_ms)


# ===================================================================
# stage 6 - z-score and the three chan_* features
# ===================================================================
class Normalizer:
    """Where the z-score and chan_* statistics come from.

    global     the notebook: the whole crop. Offline only.
    calib      frozen on the first `calib_sec`.
    expanding  from the crop start up to the current epoch.
    trailing   the last `win_sec` only.

    It matters: on DH3 the amplitude nearly triples across the first NREM
    period, so 'calib' fires ~3x too often and 'expanding' stays biased low.

    `scale` multiplies the std we divide by. A causal detrend leaves a little
    more or less low-frequency power than the notebook's, which shifts every
    feature (DH3: +7.6 % amplitude, +14.2 % energy). It has to be applied here,
    since rescaling the signal is undone by the z-score. See calib_scale().
    """

    def __init__(self, mode="global", calib_sec=300.0, scale=1.0, win_sec=600.0):
        self.mode = mode
        self.calib_sec = calib_sec
        self.scale = scale
        self.win_sec = win_sec          # 'trailing' only


def calib_scale(causal_det, ref_det, hp_hz=1.0):
    """`scale` for Normalizer, measured on a calibration block.

    Both are (n_ch, n) detrended copies of the same stretch, causal and the
    notebook's centred SMA. The centred SMA is fine there, since that block is
    already past when detection starts.
    """
    sos = scipy.signal.butter(4, hp_hz, btype="highpass", output="sos", fs=SFREQ)
    out = []
    for i in range(causal_det.shape[0]):
        a = scipy.signal.sosfiltfilt(sos, ref_det[i]).std() / ref_det[i].std()
        b = scipy.signal.sosfiltfilt(sos, causal_det[i]).std() / causal_det[i].std()
        out.append(b / a)
    return np.array(out)


def _running_moments(x, epoch_starts, w):
    """Running sums of x, x^2, x^3, x^4 at each epoch end."""
    ends = epoch_starts + w
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    c3 = np.concatenate([[0.0], np.cumsum(x ** 3)])
    c4 = np.concatenate([[0.0], np.cumsum(x ** 4)])
    n = ends.astype(float)
    return n, c1[ends], c2[ends], c3[ends], c4[ends]


def build_epochs(sig, norm):
    """z-score `sig` and cut it into 250 ms epochs.

    Returns (epochs, chan_feats). 'global' gives the notebook's constants;
    the other modes give one value per epoch.
    """
    n = len(sig)
    starts = np.arange(0, n - 4 * EPOCH, EPOCH)

    if norm.mode == "global":
        z = (sig - np.nanmean(sig)) / (np.nanstd(sig) * norm.scale)
        eps = np.stack([z[s:s + EPOCH] for s in starts])
        flat = eps.ravel()
        return eps, {"chan_ptp": np.ptp(flat), "chan_skew": sp_stats.skew(flat),
                     "chan_kurt": sp_stats.kurtosis(flat)}

    if norm.mode == "calib":
        k = int(norm.calib_sec * SFREQ)
        mu, sd = np.nanmean(sig[:k]), np.nanstd(sig[:k]) * norm.scale
        z = (sig - mu) / sd
        eps = np.stack([z[s:s + EPOCH] for s in starts])
        cal = z[:k]
        return eps, {"chan_ptp": np.ptp(cal), "chan_skew": sp_stats.skew(cal),
                     "chan_kurt": sp_stats.kurtosis(cal)}

    if norm.mode == "expanding":
        # stats over sig[0 : this epoch's end], never less than the calib block
        k = int(norm.calib_sec * SFREQ)
        ends = np.maximum(starts + EPOCH, k)
        c1 = np.concatenate([[0.0], np.cumsum(sig)])
        c2 = np.concatenate([[0.0], np.cumsum(sig ** 2)])
        mu = c1[ends] / ends
        var = np.maximum(c2[ends] / ends - mu ** 2, 1e-30)
        sd = np.sqrt(var) * norm.scale
        eps = np.stack([(sig[s:s + EPOCH] - m) / d
                        for s, m, d in zip(starts, mu, sd)])
        # chan_* over the same history
        run_max = np.maximum.accumulate(sig)[ends - 1]
        run_min = np.minimum.accumulate(sig)[ends - 1]
        c3 = np.concatenate([[0.0], np.cumsum(sig ** 3)])
        c4 = np.concatenate([[0.0], np.cumsum(sig ** 4)])
        m2 = c2[ends] / ends - mu ** 2
        m3 = c3[ends] / ends - 3 * mu * c2[ends] / ends + 2 * mu ** 3
        m4 = (c4[ends] / ends - 4 * mu * c3[ends] / ends
              + 6 * mu ** 2 * c2[ends] / ends - 3 * mu ** 4)
        return eps, {"chan_ptp": (run_max - run_min) / sd,
                     "chan_skew": m3 / np.maximum(m2, 1e-30) ** 1.5,
                     "chan_kurt": m4 / np.maximum(m2, 1e-30) ** 2 - 3.0}

    if norm.mode == "trailing":
        # stats over the last win_sec, never less than the calib block, so the
        # first epochs are not normalised by a handful of samples
        w = int(norm.win_sec * SFREQ)
        k = int(norm.calib_sec * SFREQ)
        ends = np.maximum(starts + EPOCH, k)
        lo = np.maximum(ends - w, 0)
        n = (ends - lo).astype(float)
        c1 = np.concatenate([[0.0], np.cumsum(sig)])
        c2 = np.concatenate([[0.0], np.cumsum(sig ** 2)])
        mu = (c1[ends] - c1[lo]) / n
        sd = np.sqrt(np.maximum((c2[ends] - c2[lo]) / n - mu ** 2, 1e-30)) * norm.scale
        eps = np.stack([(sig[s:s + EPOCH] - m) / d
                        for s, m, d in zip(starts, mu, sd)])
        # chan_* over the same window
        pmax = np.array([sig[a:b].max() for a, b in zip(lo, ends)])
        pmin = np.array([sig[a:b].min() for a, b in zip(lo, ends)])
        c3 = np.concatenate([[0.0], np.cumsum(sig ** 3)])
        c4 = np.concatenate([[0.0], np.cumsum(sig ** 4)])
        m2 = (c2[ends] - c2[lo]) / n - mu ** 2
        m3 = ((c3[ends] - c3[lo]) / n - 3 * mu * (c2[ends] - c2[lo]) / n
              + 2 * mu ** 3)
        m4 = ((c4[ends] - c4[lo]) / n - 4 * mu * (c3[ends] - c3[lo]) / n
              + 6 * mu ** 2 * (c2[ends] - c2[lo]) / n - 3 * mu ** 4)
        return eps, {"chan_ptp": (pmax - pmin) / sd,
                     "chan_skew": m3 / np.maximum(m2, 1e-30) ** 1.5,
                     "chan_kurt": m4 / np.maximum(m2, 1e-30) ** 2 - 3.0}

    raise ValueError(norm.mode)


# ===================================================================
# stage 7 - model and detection
# ===================================================================
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def predict_proba(feats):
    """{channel: DataFrame} -> spike probability per epoch."""
    m = model()
    e1, e2 = feats[CHANS[0]], feats[CHANS[1]]
    x = pd.concat([e1, e2], axis=1, ignore_index=True)
    x.columns = [f"eog1_{c}" for c in e1.columns] + [f"eog2_{c}" for c in e2.columns]
    return m.predict_proba(x[m.get_booster().feature_names])[:, 1]


def detect(proba, threshold=THRESHOLD, sec_limit=SEC_LIMIT, bin_length=BIN_LENGTH):
    """The notebook's stage 7: threshold crossings per bin, with a refractory
    skip. Drops a final bin under half length. Returns (times_sec, spm_per_bin).
    """
    stride = int(bin_length * SAMPLES_PER_MIN)
    refractory = int(sec_limit * 4)
    times, spm = [], []
    n = len(proba)
    for start in range(0, n, stride):
        p = proba[start:start + stride]
        if len(p) < stride // 2:
            continue
        hits = 0
        i = 0
        while i < len(p):
            if p[i] >= threshold:
                times.append(round((start + i) * EPOCH_SEC, 2))
                hits += 1
                i += refractory
            else:
                i += 1
        spm.append(hits / (len(p) / SAMPLES_PER_MIN))
    return times, spm


# ===================================================================
# the whole chain
# ===================================================================
def load_raw():
    z = np.load(RAW_CACHE, allow_pickle=True)
    return z["data"]


def crop(data):
    """Stage 2: [TMIN_MIN, TMAX_MIN] out of the cached block."""
    a = int((TMIN_MIN - CACHE_START_MIN) * 60 * SFREQ)
    b = int((TMAX_MIN - CACHE_START_MIN) * 60 * SFREQ)
    return data[:, a:b + 1]


def run(bp=None, sma_look_ms=None, sma_win_sec=SMA_SEC, norm=None,
        threshold=THRESHOLD, n_jobs=None, return_detrended=False,
        detrend=None, limit_sec=None):
    """Run the whole chain. Defaults are the notebook's.

    bp          data -> (filtered, lookahead_ms)
    detrend     filtered -> (detrended, lookahead_ms); overrides sma_*
    limit_sec   score only the first N s, for quick screens. The z-score
                still uses the whole window, as the notebook's does.
    """
    bp = bp or BandPass.mne_zero
    norm = norm or Normalizer("global")

    filt, bp_ms = bp(load_raw())
    filt = crop(filt)
    if detrend is not None:
        det, sma_ms = detrend(filt)
    else:
        det, sma_ms = detrend_sma(filt, look_ms=sma_look_ms, win_sec=sma_win_sec)
    if return_detrended:
        return det, max(bp_ms, sma_ms)

    n_ep = None if limit_sec is None else int(limit_sec * SFREQ) // EPOCH
    scales = np.atleast_1d(norm.scale).astype(float)
    feats = {}
    for i, ch in enumerate(CHANS):
        nrm = Normalizer(norm.mode, norm.calib_sec,
                         scales[i] if len(scales) > 1 else scales[0],
                         norm.win_sec)
        eps, chan = build_epochs(det[i], nrm)
        if n_ep is not None:
            eps = eps[:n_ep]
            chan = {k: (v[:n_ep] if np.ndim(v) else v) for k, v in chan.items()}
        df = epochs_to_features(eps, "DH3_Crop", n_jobs=n_jobs)
        df["chan_name"] = ch
        for k, v in chan.items():
            df[k] = v
        feats[ch] = df

    proba = predict_proba(feats)
    times, spm = detect(proba, threshold=threshold)
    return {"times": times, "spm": spm, "proba": proba,
            "lookahead_ms": max(bp_ms, sma_ms), "n_epochs": len(proba)}


# ===================================================================
# scoring against the notebook
# ===================================================================
def load_baseline_times():
    if BASELINE_JSON is not None:
        with open(BASELINE_JSON) as fh:
            return list(json.load(fh)["times"])
    import sys
    sys.path.insert(0, SIMDIR)
    from compare_notebook_vs_sim import NOTEBOOK_SEC
    return list(NOTEBOOK_SEC)


def match(a, b, tol):
    """One-to-one nearest matching within `tol` seconds."""
    cands = sorted((abs(x - y), i, j)
                   for i, x in enumerate(a) for j, y in enumerate(b)
                   if abs(x - y) <= tol)
    ua, ub, pairs = set(), set(), []
    for _, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pairs.append((i, j))
    return pairs


def score(times, ref=None, tol=0.5, limit_sec=None):
    ref = ref if ref is not None else load_baseline_times()
    if limit_sec is not None:
        # only score spikes whose whole +/-tol bracket is inside the window
        ref = [t for t in ref if t <= limit_sec - tol]
        times = [t for t in times if t <= limit_sec - tol]
    pairs = match(ref, times, tol)
    rec = len(pairs) / len(ref) if ref else 0.0
    pre = len(pairs) / len(times) if times else 0.0
    f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0
    return {"matched": len(pairs), "n_ref": len(ref), "n": len(times),
            "recall": rec, "precision": pre, "f1": f1}


if __name__ == "__main__":
    import sys
    import time
    if "--self-test" in sys.argv:
        t0 = time.time()
        r = run()
        ref = load_baseline_times()
        same = (len(r["times"]) == len(ref)
                and all(abs(x - y) < 1e-9 for x, y in zip(r["times"], ref)))
        print(f"epochs      : {r['n_epochs']} (notebook: 12237)")
        print(f"spikes      : {len(r['times'])} (notebook: {BASELINE_N_SPIKES})")
        print(f"SPM per bin : {[round(v, 2) for v in r['spm']]} (notebook: 7.4 8.6 8.1)")
        print(f"identical   : {same}")
        if not same:
            print(score(r["times"], ref, tol=0.5))
            d = [t for t in r["times"] if t not in ref]
            m = [t for t in ref if t not in r["times"]]
            print(f"extra  ({len(d)}): {d[:20]}")
            print(f"missed ({len(m)}): {m[:20]}")
        print(f"{time.time() - t0:.1f}s")
