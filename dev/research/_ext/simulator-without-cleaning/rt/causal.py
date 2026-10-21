#!/usr/bin/env python
"""
Causal stand-ins for the notebook's band-pass (stage 1) and detrend (stage 3).

Sliding the centred 2 s box back so it only reads 50 ms ahead keeps its
magnitude but not its phase: everything below ~0.5 Hz, most of the energy in
NREM, comes back at the wrong phase (r ~ 0.87 against the notebook).

So we fit causal filters to the notebook's own output instead, either by least
squares (the Wiener filter) or in the frequency domain. Fitting is done once on
a calibration block; at run time it is one convolution, and nothing reads
further ahead than the chosen lookahead.

Research code: the shipped `b25_o3` chain uses none of these designs. It is
here because `causal_chains.py` imports it.

    python causal.py           # screen the designs on DH3
    python causal.py --test    # check fit_wiener on synthetic data
"""
import os
import sys

import numpy as np
import scipy.fft
import scipy.linalg
import scipy.signal as sp_signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P                                            # noqa: E402

SFREQ = P.SFREQ


# ======================================================================
# Wiener design
# ======================================================================
def _xcorr(a, b, max_lag):
    """r[k] = sum_n a[n] b[n-k] for k = -max_lag .. max_lag, indexed [max_lag + k]."""
    n = len(a)
    nfft = scipy.fft.next_fast_len(n + max_lag)     # >= n + max_lag: no wraparound
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    full = np.fft.irfft(fa * np.conj(fb), nfft)     # full[m] = sum_n a[n] b[n-m]
    return np.concatenate([full[nfft - max_lag:], full[:max_lag + 1]])


def fit_wiener(x, y, n_past, n_future, ridge=1e-6, toeplitz=False):
    """Least-squares FIR from x to y, taps at lags -n_future .. +n_past.

    x and y: same length, aligned, centred. y can be zero outside where it is
    known. `ridge` is relative to rxx[0] and is needed: an EOG spectrum spans
    ~10 decades, so the system is badly conditioned. Cholesky by default, since
    Levinson (`toeplitz=True`) loses accuracy on such systems.

    g[0] is the most-future tap, g[-1] the oldest.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ntap = n_past + n_future + 1
    max_lag = n_past + n_future
    sx = x.std() or 1.0
    sy = y.std() or 1.0
    x = x / sx
    y = y / sy

    rxx = _xcorr(x, x, max_lag)                    # rxx[max_lag + d]
    rxy = _xcorr(y, x, max_lag)                    # sum_n y[n] x[n-k]

    col = rxx[max_lag: max_lag + ntap].copy()      # Toeplitz first column
    lags = np.arange(-n_future, n_past + 1)
    b = rxy[max_lag + lags]
    if toeplitz:
        c = col.copy()
        c[0] *= 1.0 + ridge
        g = scipy.linalg.solve_toeplitz((c, c), b)
    else:
        r = scipy.linalg.toeplitz(col)
        r[np.diag_indices_from(r)] += ridge * col[0]
        g = scipy.linalg.solve(r, b, assume_a="pos")
    return g * (sy / sx)


def apply_wiener(x, g, n_future, mean=None):
    """yhat[n] = sum_k g[k] x[n - lag_k], lag_k running -n_future .. +n_past."""
    x = np.asarray(x, float)
    x = x - (x.mean() if mean is None else mean)
    return sp_signal.fftconvolve(x, g, mode="full")[n_future: n_future + len(x)]


class WienerDetrend:
    """Fitted stage 1 + 3 replacement, one filter per channel. Use as
    `pipeline.run(bp=w, ...)`.

    calib_from="crop"    fit on the window's first `calib_sec` against the
                         notebook's output. The fit uses the future, the
                         resulting filter does not.
    calib_from="before"  fit on the `calib_sec` before the window. Causal
                         throughout.
    """

    def __init__(self, n_past, n_future, calib_sec=300.0, ridge=1e-6,
                 calib_from="crop", bp=None, bp_ms=0.0):
        self.n_past = n_past
        self.n_future = n_future
        self.calib_sec = calib_sec
        self.ridge = ridge
        self.calib_from = calib_from
        self.bp = bp                    # optional pre-filter; None = fit from raw
        self.bp_ms = bp_ms              # its own lookahead, added to the budget
        self.g = None
        self.mean = None

    def _input(self, raw):
        return raw if self.bp is None else self.bp(raw)[0]

    def fit(self, raw, target=None):
        """raw: (2, N) cache. target: (2, M) notebook-detrended crop, for 'crop'."""
        src = self._input(raw)
        a = int((P.TMIN_MIN - P.CACHE_START_MIN) * 60 * SFREQ)
        k = int(self.calib_sec * SFREQ)
        gs, means = [], []
        for ch in range(src.shape[0]):
            if self.calib_from == "before":
                # the block before the window, plus n_past of history for the fit
                lo = max(self.n_past, a - k)
                yt = _centred_detrend(src[ch, lo:a])
            else:
                lo = a
                yt = target[ch, :k]
            xs = src[ch, lo - self.n_past: lo + len(yt) + self.n_future]
            mean = xs.mean()
            xc = xs - mean
            yc = np.zeros_like(xc)
            yc[self.n_past: self.n_past + len(yt)] = yt - yt.mean()
            gs.append(fit_wiener(xc, yc, self.n_past, self.n_future, self.ridge))
            means.append(mean)
        self.g = np.array(gs)
        self.mean = np.array(means)
        return self

    def __call__(self, data):
        src = self._input(data)
        out = np.array([apply_wiener(src[i], self.g[i], self.n_future, self.mean[i])
                        for i in range(src.shape[0])])
        return out, self.bp_ms + self.n_future / SFREQ * 1000


def _centred_detrend(x, win_sec=P.SMA_SEC):
    w = int(SFREQ * win_sec)
    k = np.ones(w) / w
    return x - np.convolve(x, k, mode="same")


# ======================================================================
# frequency-domain causal design
# ======================================================================
def _wrap(taps, lags, nfft):
    """Put taps at signed lags into an nfft buffer, y[n] = sum t[k] x[n-k]."""
    out = np.zeros(nfft)
    np.add.at(out, np.asarray(lags) % nfft, taps)
    return out


def notebook_response(nfft=1 << 16, win_sec=P.SMA_SEC, bp=True):
    """Stage 1 + 3 frequency response, as (freqs, H), from the real kernels.

    The box has even length, so mode='same' centres it half a sample off (1000
    taps back, 999 ahead). Both kernels are near-symmetric, so H is real.
    """
    f = np.fft.rfftfreq(nfft, 1 / SFREQ)
    w = int(SFREQ * win_sec)
    # y[n] = mean(x[n-w/2 .. n+w/2-1]) -> lags -(w/2 - 1) .. +w/2
    lags = np.arange(-(w // 2 - 1), w // 2 + 1)
    t = _wrap(-np.ones(w) / w, lags, nfft)
    t[0] += 1.0                                        # delta - box
    h = np.fft.rfft(t)
    if bp:
        k = P._mne_kernel()
        c = (len(k) - 1) // 2
        h = h * np.fft.rfft(_wrap(k, np.arange(-c, c + 1), nfft))
    return f, h


def design_causal(n_taps, delay, psd=None, nfft=1 << 16, floor_db=60.0,
                  win_sec=P.SMA_SEC, bp=True):
    """Causal FIR of `n_taps` matching stage 1 + 3 delayed by `delay` samples,
    weighted by `psd`.

    The Wiener filter again, but against a smoothed spectrum floored at
    `floor_db` below its peak. Without the floor the time-domain equations are
    singular in practice (the EOG PSD spans ~12 decades) and the fit is noise.

    Returns (h, delay). Apply as y[n] = (h * x)[n + delay].
    """
    f, hd = notebook_response(nfft, win_sec=win_sec, bp=bp)
    if psd is None:
        s = np.ones_like(f)
    else:
        s = np.maximum(psd, psd.max() * 10 ** (-floor_db / 10))
    target = hd * np.exp(-2j * np.pi * f * delay / SFREQ)

    # minimise int S |H - target|^2  ->  R h = p,
    #   R[i,j] = int S e^{-j2pi f (j-i)} = r_s(i-j),  p[i] = int S target e^{+j2pi f i}
    rs = np.fft.irfft(s, nfft)[:n_taps]
    p = np.fft.irfft(s * target, nfft)[:n_taps]
    r = scipy.linalg.toeplitz(rs)
    r[np.diag_indices_from(r)] += 1e-12 * rs[0]
    h = scipy.linalg.solve(r, p, assume_a="pos")
    return h, delay


def channel_psd(x, nfft=1 << 16):
    """Welch PSD of x, on notebook_response's grid."""
    f, pxx = sp_signal.welch(x, SFREQ, nperseg=nfft, noverlap=nfft // 2,
                             nfft=nfft, return_onesided=True)
    return pxx


class CausalStage13:
    """Stage 1 + 3 as one causal FIR per channel. Use as pipeline `bp=`.

    The target response is the notebook's; only the weighting comes from the
    calibration block's spectrum, and it only has to be roughly right.
    """

    def __init__(self, n_taps=8192, delay=50, calib_sec=300.0, floor_db=60.0,
                 win_sec=P.SMA_SEC, calib_from="crop"):
        self.n_taps = n_taps
        self.delay = delay
        self.calib_sec = calib_sec
        self.floor_db = floor_db
        self.win_sec = win_sec
        self.calib_from = calib_from
        self.h = None
        self.mean = None

    def fit(self, raw):
        a = int((P.TMIN_MIN - P.CACHE_START_MIN) * 60 * SFREQ)
        k = int(self.calib_sec * SFREQ)
        lo, hi = (a, a + k) if self.calib_from == "crop" else (max(0, a - k), a)
        hs, means = [], []
        for ch in range(raw.shape[0]):
            seg = raw[ch, lo:hi]
            h, _ = design_causal(self.n_taps, self.delay,
                                 psd=channel_psd(seg - seg.mean()),
                                 floor_db=self.floor_db, win_sec=self.win_sec)
            hs.append(h)
            means.append(seg.mean())
        self.h = np.array(hs)
        self.mean = np.array(means)
        return self

    def __call__(self, data):
        out = np.array([
            sp_signal.fftconvolve(data[i] - self.mean[i], self.h[i],
                                  mode="full")[self.delay: self.delay + data.shape[1]]
            for i in range(data.shape[0])])
        return out, self.delay / SFREQ * 1000


# ======================================================================
# stage 3 replacements, as pipeline `detrend=` callables
# ======================================================================
# The notebook's detrend is a zero-phase high-pass with a ~0.32 Hz corner. With
# 50 ms of lookahead its waveform cannot be matched (full-band r caps at ~0.88),
# but everything above ~1 Hz, where the spikes are, can (r ~ 0.99). So these
# aim at what the model reacts to: how much sub-0.5 Hz power is left (notebook
# 9 %, shifted box 40 %, which shrinks every amplitude feature through the
# z-score) and the scale of the spike band.

def sma_detrend(look_ms):
    """The notebook's box, slid forward. The baseline to beat."""
    def f(x):
        return P.detrend_sma(x, look_ms=look_ms)
    return f


def butter_hp(order, fc, look_ms=0.0):
    """Causal Butterworth high-pass, no lookahead."""
    sos = sp_signal.butter(order, fc, btype="highpass", output="sos", fs=SFREQ)
    adv = int(round(look_ms / 1000 * SFREQ))

    def f(x):
        y = np.array([sp_signal.sosfilt(sos, ch - ch.mean()) for ch in x])
        if adv:
            y = np.concatenate([y[:, adv:], np.zeros((y.shape[0], adv))], axis=1)
        return y, look_ms
    return f


def minphase_from_magnitude(mag, n_taps, nfft=1 << 16):
    """Minimum-phase FIR with magnitude `mag` (rfft grid), via the cepstrum.

    The least phase lag any causal filter with this magnitude can have. Not
    scipy.signal.minimum_phase: before scipy 1.14 that returns the square root
    of the magnitude, which leaves 17 % of the power below 0.5 Hz instead of 9 %.
    """
    m = np.maximum(np.abs(mag), 1e-12)
    c = np.fft.irfft(np.log(m), nfft)                  # real cepstrum
    c[1:nfft // 2] *= 2.0
    c[nfft // 2 + 1:] = 0.0
    h = np.fft.irfft(np.exp(np.fft.rfft(c)), nfft)
    return h[:n_taps]


def minphase_detrend(extra_hp=None, n_taps=4096, look_ms=0.0,
                     win_sec=P.SMA_SEC, nfft=1 << 16):
    """Causal detrend with the notebook's magnitude response, minimum phase.

    The notebook's box detrend overshoots (+21 % at 0.75 Hz, +13 % at 1.25 Hz),
    where most of the power is, so a Butterworth gets the delta band wrong.

    `extra_hp=(order, fc)` multiplies in a further high-pass: below it no causal
    filter gets the phase right, so removing that 9 % of power beats adding it
    back at the wrong phase.
    """
    f, hd = notebook_response(nfft, win_sec=win_sec, bp=False)
    mag = np.abs(hd)
    if extra_hp is not None:
        order, fc = extra_hp
        b, a = sp_signal.butter(order, fc, btype="highpass", fs=SFREQ)
        mag = mag * np.abs(sp_signal.freqz(b, a, worN=f, fs=SFREQ)[1])
    h = minphase_from_magnitude(mag, n_taps, nfft)
    adv = int(round(look_ms / 1000 * SFREQ))

    def fn(x):
        y = np.array([sp_signal.fftconvolve(ch - ch.mean(), h,
                                            mode="full")[:x.shape[1]] for ch in x])
        if adv:
            y = np.concatenate([y[:, adv:], np.zeros((y.shape[0], adv))], axis=1)
        return y, look_ms
    fn.h = h
    return fn


def wiener_detrend(delay=50, n_taps=4096, weight="psd", alpha=1.0,
                   floor_db=60.0, calib_sec=300.0, win_sec=P.SMA_SEC,
                   extra_hp=None, nfft=1 << 16):
    """Best causal detrend for a given lookahead of `delay` samples.

    Fits the FIR closest to the notebook's detrend delayed by `delay`: 0 is
    near minimum phase, 1000 reproduces the notebook (r = 1.000). Sweeping it
    gives accuracy against latency.

    weight='psd' weights by the signal's spectrum (output MSE); 'flat' gets the
    sub-1 Hz gain wrong and the output 5x too large. alpha: psd ** alpha.

    `extra_hp=(order, fc)` rolls off what the delay cannot phase-match. On DH3
    the model coped far better with missing slow content (recall 0.89) than
    with slow content at the wrong phase (0.19).
    """
    f, hd = notebook_response(nfft, win_sec=win_sec, bp=False)
    if extra_hp is not None:
        b_, a_ = sp_signal.butter(extra_hp[0], extra_hp[1], btype="highpass", fs=SFREQ)
        hd = hd * np.abs(sp_signal.freqz(b_, a_, worN=f, fs=SFREQ)[1])

    def build(psd):
        if weight == "flat" or psd is None:
            s = np.ones_like(f)
        else:
            s = np.maximum(psd, psd.max() * 10 ** (-floor_db / 10)) ** alpha
            s = s / s.max()
        target = hd * np.exp(-2j * np.pi * f * delay / SFREQ)
        rs = np.fft.irfft(s, nfft)[:n_taps]
        p = np.fft.irfft(s * target, nfft)[:n_taps]
        r = scipy.linalg.toeplitz(rs)
        r[np.diag_indices_from(r)] += 1e-10 * rs[0]
        return scipy.linalg.solve(r, p, assume_a="pos")

    state = {}

    def fn(x):
        if not state:
            k = int(calib_sec * SFREQ)
            state["h"] = [build(None if weight == "flat"
                                else channel_psd(x[i, :k] - x[i, :k].mean()))
                          for i in range(x.shape[0])]
        y = np.array([
            sp_signal.fftconvolve(x[i] - x[i].mean(), state["h"][i],
                                  mode="full")[delay: delay + x.shape[1]]
            for i in range(x.shape[0])])
        return y, delay / SFREQ * 1000
    fn.state = state
    return fn


def minphase_hp(win_sec=P.SMA_SEC, look_ms=0.0):
    """Deprecated, use minphase_detrend(). Kept so an old measurement can be
    rerun. Hits the scipy < 1.14 square-root bug described there."""
    w = int(SFREQ * win_sec) | 1
    d = -np.ones(w) / w
    d[w // 2] += 1.0
    h = sp_signal.minimum_phase(d, method="hilbert")
    adv = int(round(look_ms / 1000 * SFREQ))

    def f(x):
        y = np.array([sp_signal.lfilter(h, 1.0, ch - ch.mean()) for ch in x])
        if adv:
            y = np.concatenate([y[:, adv:], np.zeros((y.shape[0], adv))], axis=1)
        return y, look_ms
    return f


def _box_dft(win_sec, nfft):
    """K(f) of the length-w box, on the rfft grid."""
    w = int(SFREQ * win_sec)
    k = np.zeros(nfft)
    k[:w] = 1.0 / w
    return np.fft.rfft(k)


def box_corrector(look_ms=50.0, win_sec=P.SMA_SEC, n_taps=2048, nfft=1 << 16,
                  floor_db=60.0, extra_hp=1.0):
    """Minimum-phase FIR that undoes what sliding the box forward breaks.

    Above 0.5 Hz the slid box already matches (r = 0.997 over 1-40 Hz). Below
    it the response comes out up to 9x too tall, and through the z-score that
    shrinks every amplitude feature. So we correct only the magnitude there:

        C(f) = |1 - K e^{j w 1000}| / |1 - K e^{j w look}|

    Minimum phase, so no extra lookahead, and its delay sits below 0.5 Hz.
    `extra_hp` raises C to a power: 1 matches the notebook, >1 removes more.
    """
    f = np.fft.rfftfreq(nfft, 1 / SFREQ)
    k = _box_dft(win_sec, nfft)
    w = 2 * np.pi * f / SFREQ
    centred = np.abs(1 - k * np.exp(1j * w * (int(SFREQ * win_sec) // 2)))
    shifted = np.abs(1 - k * np.exp(1j * w * int(round(look_ms / 1000 * SFREQ))))
    c = centred / np.maximum(shifted, shifted.max() * 10 ** (-floor_db / 20))
    if extra_hp != 1.0:
        c = c ** extra_hp
    return minphase_from_magnitude(np.minimum(c, 1.0), n_taps, nfft)


def box_plus_corrector(look_ms=50.0, win_sec=P.SMA_SEC, n_taps=2048,
                       extra_hp=1.0):
    """Stage 3 replacement: the box slid to `look_ms`, then box_corrector.
    Lookahead is `look_ms`. For `pipeline.run(detrend=...)`."""
    c = box_corrector(look_ms, win_sec, n_taps, extra_hp=extra_hp)

    def f(x):
        y, _ = P.detrend_sma(x, look_ms=look_ms, win_sec=win_sec)
        y = np.array([sp_signal.lfilter(c, 1.0, ch) for ch in y])
        return y, look_ms
    return f


def rescaled(inner, ref_hp_hz=1.0):
    """Deprecated and a no-op: scaling the signal is undone by the z-score.
    Use pipeline.Normalizer(scale=...), which scales the divisor instead."""
    sos = sp_signal.butter(4, ref_hp_hz, btype="highpass", output="sos", fs=SFREQ)

    def f(x):
        y, ms = inner(x)
        ref, _ = P.detrend_sma(x, look_ms=None)
        for i in range(y.shape[0]):
            a = sp_signal.sosfiltfilt(sos, ref[i]).std()
            b = sp_signal.sosfiltfilt(sos, y[i]).std()
            y[i] *= a / b
        return y, ms
    return f


# ======================================================================
# screening
# ======================================================================
def notebook_target(raw):
    filt, _ = P.BandPass.mne_zero(raw)
    det, _ = P.detrend_sma(P.crop(filt), look_ms=None)
    return det


def r_vs(ref, cand, skip=30_000):
    sl = slice(skip, ref.shape[1] - skip)
    out = []
    for i in range(ref.shape[0]):
        a = ref[i][sl]
        b = cand[i][sl]
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        out.append(float(np.dot(a, b) / len(a)))
    return out


def selftest():
    """fit_wiener has to recover a known non-causal FIR from its own output."""
    rng = np.random.default_rng(0)
    n_past, n_future = 40, 7
    g_true = rng.normal(size=n_past + n_future + 1) * np.exp(
        -np.arange(n_past + n_future + 1) / 12)
    x = sp_signal.lfilter([1.0], [1.0, -0.97], rng.normal(size=200_000))  # 1/f-ish
    y = sp_signal.fftconvolve(x, g_true, mode="full")[n_future: n_future + len(x)]
    g = fit_wiener(x - x.mean(), y - y.mean(), n_past, n_future, ridge=1e-12)
    err = np.abs(g - g_true).max() / np.abs(g_true).max()
    yhat = apply_wiener(x, g, n_future)
    r = np.corrcoef(y[n_past:-n_future], yhat[n_past:-n_future])[0, 1]
    print(f"selftest: max coef error {err:.2e}, r {r:.8f}")
    # the coefficients carry a small edge bias, the reconstruction does not
    assert err < 1e-2 and r > 1 - 1e-8, "fit_wiener does not recover a known filter"


def main():
    raw = P.load_raw()
    ref = notebook_target(raw)
    print(f"target = notebook stage1+stage3 over {P.TMIN_MIN}-{P.TMAX_MIN} min\n")

    print(f"{'design':52s} {'look':>5s} | {'r E227':>8s} {'r E254':>8s}")
    print("-" * 80)

    # baseline: the shifted box
    for look in (25, 50):
        filt, _ = P.BandPass.firwin(raw, 51, False)
        det, _ = P.detrend_sma(P.crop(filt), look_ms=look)
        r = r_vs(ref, det)
        print(f"{'shifted box: firwin51 + SMA ' + str(look) + 'ms':52s} "
              f"{look + 25:5d} | {r[0]:8.4f} {r[1]:8.4f}")
    print()

    for n_future in (25, 50):
        for n_past in (2000, 4000, 8000, 16000):
            w = WienerDetrend(n_past, n_future, 300.0).fit(raw, ref)
            cand = P.crop(w(raw)[0])
            r = r_vs(ref, cand)
            dc = np.abs(w.g.sum(axis=1)).max()
            print(f"{'wiener raw->detrended  past=' + str(n_past):52s} "
                  f"{n_future:5d} | {r[0]:8.4f} {r[1]:8.4f}   |sum g|<={dc:.2e}")
        print()

    # fully causal: fit on [22, 23) min, before the window opens
    for n_future in (25, 50):
        for n_past in (4000, 8000):
            w = WienerDetrend(n_past, n_future, 60.0, calib_from="before").fit(raw)
            cand = P.crop(w(raw)[0])
            r = r_vs(ref, cand)
            print(f"{'wiener fit on [22,23) min  past=' + str(n_past):52s} "
                  f"{n_future:5d} | {r[0]:8.4f} {r[1]:8.4f}")
        print()


if __name__ == "__main__":
    if "--test" in sys.argv:
        selftest()
    else:
        main()


def compose(*fns):
    """Chain detrend callables; their lookaheads add up."""
    def fn(x):
        ms = 0.0
        for f in fns:
            x, m = f(x)
            ms += m
        return x, ms
    return fn
