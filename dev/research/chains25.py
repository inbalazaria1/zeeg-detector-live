#!/usr/bin/env python
"""
Causal band-pass + detrend chains with at most 25 ms of lookahead.

Each chain is `name -> (bandpass_fn, detrend_fn, look_ms, note)`:

    stage 1a   causal Butterworth 0.1 Hz high-pass (sosfilt)     0 ms
    stage 1b   symmetric firwin 70 Hz low-pass, read centred      (N-1)/2 ms
    stage 3    trailing box detrend                               0 ms

An odd N-tap low-pass costs (N-1)/2 ms, so N <= 51. The DC offset comes from
the calibration block only, so nothing reads past `look_ms`.

Names are `_`-joined tokens, any order, defaults in brackets:
o{N} high-pass order [3], t{N} taps [51], w{name} window [hamming],
k{beta} kaiser beta [8.6], lp{f} cutoff Hz [70], box{s} box length s [2.0].
`b25_o3` = `o3_t51`, the chain the detector ships with.
"""
import os
import sys

import numpy as np
import scipy.signal as sp_signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "_ext")
for _p in (os.path.join(ROOT, "bin20"),
           os.path.join(ROOT, "simulator-without-cleaning", "rt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as P                                             # noqa: E402
import causal_chains as CC                                       # noqa: E402

LOOK_BUDGET_MS = 25.0
MAX_TAPS = 51
WINDOWS = ("hamming", "hann", "blackman", "kaiser")
DEFAULT_KAISER_BETA = 8.6


# ======================================================================
# the spec
# ======================================================================
class ChainSpec:
    """One (high-pass order, low-pass design, box length) choice."""

    def __init__(self, hp_order=3, ntaps=MAX_TAPS, window="hamming",
                 beta=DEFAULT_KAISER_BETA, lp_hz=P.BP_HIGH, box_sec=2.0,
                 hp_hz=P.BP_LOW):
        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (symmetric kernel): {ntaps}")
        if window not in WINDOWS:
            raise ValueError(f"unknown window {window!r}, want one of {WINDOWS}")
        self.hp_order = int(hp_order)
        self.ntaps = int(ntaps)
        self.window = window
        self.beta = float(beta)
        self.lp_hz = float(lp_hz)
        self.box_sec = float(box_sec)
        self.hp_hz = float(hp_hz)
        if self.look_ms() > LOOK_BUDGET_MS + 1e-9:
            raise ValueError(
                f"{self.name()} costs {self.look_ms():g} ms of lookahead, over "
                f"this folder's {LOOK_BUDGET_MS:g} ms ceiling "
                f"(an odd N-tap symmetric low-pass costs (N-1)/2 ms, so "
                f"N <= {MAX_TAPS})")

    # -- identity ------------------------------------------------------
    def look_ms(self):
        return (self.ntaps - 1) // 2 / P.SFREQ * 1000.0

    def name(self):
        s = f"o{self.hp_order}_t{self.ntaps}"
        if self.window != "hamming":
            s += f"_w{self.window}"
            if self.window == "kaiser":
                s += f"_k{self.beta:g}"
        if abs(self.lp_hz - P.BP_HIGH) > 1e-12:
            s += f"_lp{self.lp_hz:g}"
        if abs(self.box_sec - 2.0) > 1e-12:
            s += f"_box{self.box_sec:g}"
        return s

    def note(self):
        w = self.window + (f"(beta {self.beta:g})" if self.window == "kaiser"
                           else "")
        return (f"IIR butter o{self.hp_order} {self.hp_hz:g} Hz HP + "
                f"{self.ntaps}-tap {w} {self.lp_hz:g} Hz LP "
                f"({self.look_ms():g} ms) + trailing {self.box_sec:g} s box")

    def __repr__(self):
        return f"ChainSpec({self.name()})"

    # -- the blocks ----------------------------------------------------
    def fir_window(self):
        return (("kaiser", self.beta) if self.window == "kaiser"
                else self.window)

    def lp_kernel(self):
        # hamming is firwin's default; calling it without `window` keeps the
        # kernel bit-identical to the one the shipped matrices were built with
        if self.window == "hamming":
            return sp_signal.firwin(self.ntaps, self.lp_hz, fs=P.SFREQ)
        return sp_signal.firwin(self.ntaps, self.lp_hz,
                                window=self.fir_window(), fs=P.SFREQ)

    def hp_sos(self):
        return sp_signal.butter(self.hp_order, self.hp_hz, btype="highpass",
                                output="sos", fs=P.SFREQ)

    def bandpass(self):
        """High-pass, then the centred low-pass. Returns `f(data)`."""
        def f(d):
            k = CC.calib_end_raw()
            h = self.lp_kernel()
            sos = self.hp_sos()
            out = [np.convolve(sp_signal.sosfilt(sos, CC._demean(ch, k)), h,
                               mode="same") for ch in d]
            return np.array(out), self.look_ms()
        return f

    def detrend(self):
        """Trailing box detrend, no lookahead."""
        return CC.trail_box(self.box_sec)

    def tuple(self):
        return (self.bandpass(), self.detrend(), self.look_ms(), self.note())


# ======================================================================
# names
# ======================================================================
def parse(name):
    """`'o4_t31_whann'` -> ChainSpec.  Tokens may appear in any order."""
    name = ALIASES.get(name, name)
    kw = {}
    toks = [t for t in name.split("_") if t]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("box"):
            kw["box_sec"] = float(t[3:])
        elif t.startswith("lp"):
            kw["lp_hz"] = float(t[2:])
        elif t.startswith("o") and t[1:].isdigit():
            kw["hp_order"] = int(t[1:])
        elif t.startswith("t") and t[1:].isdigit():
            kw["ntaps"] = int(t[1:])
        elif t.startswith("w"):
            kw["window"] = t[1:]
        elif t.startswith("k"):
            kw["beta"] = float(t[1:])
        else:
            raise ValueError(f"unparsable chain token {t!r} in {name!r}")
        i += 1
    return ChainSpec(**kw)


def make(hp_order=3, ntaps=MAX_TAPS, window="hamming",
         beta=DEFAULT_KAISER_BETA, lp_hz=P.BP_HIGH, box_sec=2.0):
    """`parse` with keyword arguments instead of a name."""
    return ChainSpec(hp_order=hp_order, ntaps=ntaps, window=window, beta=beta,
                     lp_hz=lp_hz, box_sec=box_sec)


# the shipped chain keeps its own name so results are always labelled the same
ALIASES = {"b25_o3": "o3_t51"}

# chains built so far; any name `parse()` accepts works, registered or not
CHAINS = {}


def _register(name):
    spec = parse(name)
    CHAINS[name] = spec.tuple()
    return spec


def spec(name):
    """ChainSpec for `name` (registered or freshly parsed)."""
    return parse(name)


def get(name):
    """(bandpass_fn, detrend_fn, look_ms, note) for `name`."""
    if name not in CHAINS:
        CHAINS[name] = parse(name).tuple()
    return CHAINS[name]


def note(name):
    return parse(name).note()


def look_ms(name):
    return parse(name).look_ms()


# the shipped chain plus the variants that were compared against it
for _n in (["b25_o3"]
           + [f"o{o}_t51" for o in (2, 3, 4, 5, 6)]
           + [f"o3_t{t}" for t in (11, 21, 31, 41, 51)]
           + [f"o3_t51_w{w}" for w in ("hann", "blackman")]
           + ["o3_t51_wkaiser_k8.6"]):
    _register(_n)


if __name__ == "__main__":
    for n in CHAINS:
        s = parse(n)
        print(f"{n:<24s} {s.look_ms():5.1f} ms   {s.note()}")
