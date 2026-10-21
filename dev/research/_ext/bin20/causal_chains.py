#!/usr/bin/env python
"""
Causal stage-1 and stage-3 filters that also take their mean causally.

The filters in `pipeline.py` / `causal.py` read only `look_ms` of future, but
first subtract a mean over the whole array.  We measured that as harmless,
but here the mean comes from the calibration block only, so with the expanding
z-score a chain reads nothing past its `look_ms`.

Lookahead of each:

    trunc_causal(50)     50 ms      firwin_causal(101)   50 ms
    firwin_causal(51)    25 ms      iir_causal(10)        0 ms
    trail_box            0 ms       minph_causal          0 ms
    butter_causal        0 ms
"""
import os
import sys

import numpy as np
import scipy.signal as sp_signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "simulator-without-cleaning", "rt"))

import subjects as S                                            # noqa: E402
import pipeline as P                                            # noqa: E402
import causal as C                                              # noqa: E402


def calib_end_raw():
    """Index into the raw cache where the calibration block ends.

    The cache starts before the window (filter warm-up), so this counts the
    warm-up plus the 20 min calibration block.
    """
    crop0 = int((P.TMIN_MIN - P.CACHE_START_MIN) * 60 * P.SFREQ)
    return crop0 + int(S.CALIB_SEC * P.SFREQ)


def calib_end_crop():
    """The same boundary, counted from the start of the crop."""
    return int(S.CALIB_SEC * P.SFREQ)


def _demean(ch, k):
    return ch - ch[:k].mean()


# ======================================================================
# stage 1
# ======================================================================
def trunc_causal(look_ms):
    """The notebook's kernel with future taps past `look_ms` dropped.

    Rescaled to unit gain over 1-40 Hz.  Cutting the taps loses the 0.1 Hz
    corner at DC (taps sum to 0.489), which is why the mean has to be causal.
    """
    def f(d):
        k = calib_end_raw()
        h = P._mne_kernel()
        c = (len(h) - 1) // 2
        look = int(round(look_ms / 1000 * P.SFREQ))
        hh = h[c - look:].copy()
        w, hf = sp_signal.freqz(hh, worN=4096, fs=P.SFREQ)
        hh /= np.abs(hf[(w >= 1) & (w <= 40)]).mean()
        out = np.array([
            sp_signal.fftconvolve(_demean(ch, k), hh, mode="full")[look:look + len(ch)]
            for ch in d])
        return out, float(look_ms)
    return f


def firwin_causal(ntaps):
    """Short symmetric 70 Hz FIR applied centred, after a causal 0.1 Hz IIR.

    Costs (ntaps - 1) // 2 samples: 51 taps 25 ms, 101 taps 50 ms.  A 0.1 Hz
    corner does not fit in a short FIR, so it is an order-2 Butterworth.
    """
    def f(d):
        k = calib_end_raw()
        h = sp_signal.firwin(ntaps, P.BP_HIGH, fs=P.SFREQ)
        sos = sp_signal.butter(2, P.BP_LOW, btype="highpass", output="sos",
                               fs=P.SFREQ)
        out = [np.convolve(sp_signal.sosfilt(sos, _demean(ch, k)), h, mode="same")
               for ch in d]
        return np.array(out), (ntaps - 1) // 2 / P.SFREQ * 1000
    return f


def iir_causal(order=10):
    """Causal Butterworth 0.1-70 Hz band-pass, no lookahead."""
    def f(d):
        k = calib_end_raw()
        sos = sp_signal.butter(order, [P.BP_LOW, P.BP_HIGH], btype="bandpass",
                               output="sos", fs=P.SFREQ)
        return np.array([sp_signal.sosfilt(sos, _demean(ch, k)) for ch in d]), 0.0
    return f


# ======================================================================
# stage 3
# ======================================================================
def trail_box(win_sec=2.0):
    """The notebook's 2 s box, trailing only.  It removes its own mean."""
    def f(x):
        return P.detrend_sma(x, look_ms=0.0, win_sec=win_sec)
    return f


def minph_causal(win_sec=P.SMA_SEC):
    """Minimum-phase FIR with the same magnitude response as the notebook's box.

    The kernel from `causal.minphase_detrend`, mean from the calibration block.
    """
    h = C.minphase_detrend(None, win_sec=win_sec).h

    def f(x):
        k = calib_end_crop()
        return np.array([
            sp_signal.fftconvolve(_demean(ch, k), h, mode="full")[:x.shape[1]]
            for ch in x]), 0.0
    return f


def butter_causal(order=3, fc=0.4):
    """A causal Butterworth high-pass in place of the box."""
    def f(x):
        k = calib_end_crop()
        sos = sp_signal.butter(order, fc, btype="highpass", output="sos",
                               fs=P.SFREQ)
        return np.array([sp_signal.sosfilt(sos, _demean(ch, k)) for ch in x]), 0.0
    return f


def box_then_hp_causal(win_sec=2.0, order=3, fc=0.4):
    """The trailing box, then a causal high-pass on top."""
    hp = butter_causal(order, fc)

    def f(x):
        y, _ = P.detrend_sma(x, look_ms=0.0, win_sec=win_sec)
        return hp(y)[0], 0.0
    return f
