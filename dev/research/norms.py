#!/usr/bin/env python
"""
Stage 6 z-score variants, all causal. The detector ships with `expand20`.

    build_epochs(sig, spec) -> (epochs, {"chan_ptp", "chan_skew", "chan_kurt"})

`epochs` is (n_epochs, 250) of z-scored signal, and the `chan_*` arrays come
from the same statistics as the z-score. Nothing reads past the end of the
epoch being decided.

Specs are `+`-joined: one base mode, then any modifiers.

    expand20     expanding, seeded on the 20 min calibration block. Bit-identical
                 to `pipeline.build_epochs(sig, Normalizer("expanding"))`.
    expandw{W}   expanding, but the seed counts as W minutes instead of 20.
                 W < 20 lets the live signal take over sooner.
    ewma{T}      exponentially weighted, time constant T minutes.
    trail{W}     trailing W minutes, never shorter than the calibration block.
    +wins{c}     clip samples to +-c sd before they enter the statistics.
    +sc{c}       multiply the z-score divisor by a constant c.

e.g. `expandw5+wins6`, `expand20+sc1.3`.

* ewma decays once per 250 ms epoch, which is how a device would update it.
  T minutes of memory means an effective count of T*60*SFREQ samples.
* `wins` clips each epoch against the running estimate through the previous
  epoch, not a band frozen on the calibration block: on DH3 the amplitude
  nearly triples across a NREM period, so a frozen band would clip more and
  more real signal. Only the statistics see the clipped signal; the epoch that
  gets z-scored is the raw one, but `chan_ptp` uses the clipped extremes.
"""
import os
import sys

import numpy as np
import scipy.stats as sp_stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "_ext")
for _p in (os.path.join(ROOT, "bin20"),
           os.path.join(ROOT, "simulator-without-cleaning", "rt")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as P                                             # noqa: E402
import subjects as S                                             # noqa: E402

EPOCH = P.EPOCH
SFREQ = P.SFREQ
TINY = 1e-30


# ======================================================================
# the spec
# ======================================================================
class NormSpec:
    """One stage 6 choice. `parse()` builds it from a name.

    mode      'expand' | 'ewma' | 'trail'
    seed_w    expand: seed weight in minutes, None = its true length
    tau_min   ewma: time constant in minutes
    win_min   trail: window in minutes
    wins_c    None, or the clip half-width in sd
    scale     multiplier on the z-score divisor (1.0 = none)
    calib_sec calibration length, default `subjects.CALIB_SEC`
    """

    def __init__(self, mode="expand", seed_w=None, tau_min=None, win_min=None,
                 wins_c=None, calib_sec=None, scale=1.0):
        if mode not in ("expand", "ewma", "trail"):
            raise ValueError(f"unknown norm mode {mode!r}")
        if mode == "ewma" and not tau_min:
            raise ValueError("ewma needs a time constant, e.g. ewma30")
        if mode == "trail" and not win_min:
            raise ValueError("trail needs a window, e.g. trail20")
        if float(scale) <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.mode = mode
        self.seed_w = None if seed_w is None else float(seed_w)
        self.tau_min = None if tau_min is None else float(tau_min)
        self.win_min = None if win_min is None else float(win_min)
        self.wins_c = None if wins_c is None else float(wins_c)
        self._calib_sec = calib_sec
        self.scale = float(scale)

    @property
    def calib_sec(self):
        return S.CALIB_SEC if self._calib_sec is None else self._calib_sec

    def name(self):
        if self.mode == "expand":
            base = "expand20" if self.seed_w is None else f"expandw{self.seed_w:g}"
        elif self.mode == "ewma":
            base = f"ewma{self.tau_min:g}"
        else:
            base = f"trail{self.win_min:g}"
        if self.wins_c is not None:
            base += f"+wins{self.wins_c:g}"
        if abs(self.scale - 1.0) > 1e-12:
            base += f"+sc{self.scale:g}"
        return base

    def __repr__(self):
        return f"NormSpec({self.name()})"

    def __str__(self):
        return self.name()


def parse(name):
    """`'expandw5+wins6'` -> NormSpec."""
    if isinstance(name, NormSpec):
        return name
    toks = [t for t in str(name).split("+") if t]
    if not toks:
        raise ValueError("empty norm spec")
    base, mods = toks[0], toks[1:]
    kw = {}
    if base in ("expand20", "expand", "expanding"):
        kw.update(mode="expand", seed_w=None)
    elif base.startswith("expandw"):
        kw.update(mode="expand", seed_w=float(base[len("expandw"):]))
    elif base.startswith("ewma"):
        kw.update(mode="ewma", tau_min=float(base[len("ewma"):]))
    elif base.startswith("trail"):
        kw.update(mode="trail", win_min=float(base[len("trail"):]))
    else:
        raise ValueError(f"unknown norm base mode {base!r}")
    for m in mods:
        if m.startswith("wins"):
            kw["wins_c"] = float(m[len("wins"):])
        elif m.startswith("sc"):
            kw["scale"] = float(m[len("sc"):])
        else:
            raise ValueError(f"unknown norm modifier {m!r}")
    return NormSpec(**kw)


# ======================================================================
# helpers
# ======================================================================
def _epoch_grid(n, k):
    """Epoch starts, and where each epoch's statistics end, as in `pipeline`."""
    starts = np.arange(0, n - 4 * EPOCH, EPOCH)
    ends = np.maximum(starts + EPOCH, k)
    return starts, ends


def _cums(x):
    """Prefix sums of x, x^2, x^3, x^4 (length n+1, c[0] = 0)."""
    return [np.concatenate([[0.0], np.cumsum(x ** p)]) for p in (1, 2, 3, 4)]


def _from_raw_moments(s1, s2, s3, s4, n, scale, pmax, pmin):
    """(mu, sd, chan_*) from raw power sums and an effective sample count."""
    mu = s1 / n
    m2 = s2 / n - mu ** 2
    m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
    m4 = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
    sd = np.sqrt(np.maximum(m2, TINY)) * scale
    chan = {"chan_ptp": (pmax - pmin) / sd,
            "chan_skew": m3 / np.maximum(m2, TINY) ** 1.5,
            "chan_kurt": m4 / np.maximum(m2, TINY) ** 2 - 3.0}
    return mu, sd, chan


def _cut(sig, starts, mu, sd):
    return np.stack([(sig[s:s + EPOCH] - m) / d
                     for s, m, d in zip(starts, mu, sd)])


# ======================================================================
# winsorising: build the stream the accumulator actually sees
# ======================================================================
def _winsorise(sig, spec, starts, ends, k):
    """The clipped signal, same length as `sig`.

    The calibration block is clipped against its own mean / sd, each later
    epoch against the running estimate through the previous epoch. Has to be a
    loop for that reason.
    """
    c = spec.wins_c
    xc = np.empty_like(sig, dtype=float)
    mu0, sd0 = float(np.nanmean(sig[:k])), float(np.nanstd(sig[:k]))
    sd0 = max(sd0, TINY)
    xc[:k] = np.clip(sig[:k], mu0 - c * sd0, mu0 + c * sd0)

    n = len(sig)
    # prefix sums of the clipped signal, filled as we go (ewma keeps its own)
    c1 = np.zeros(n + 1)
    c2 = np.zeros(n + 1)
    c1[1:k + 1] = np.cumsum(xc[:k])
    c2[1:k + 1] = np.cumsum(xc[:k] ** 2)

    kk = float(k)
    m_seed = (kk if spec.seed_w is None
              else max(spec.seed_w * 60.0 * SFREQ, 1.0))
    seed_r = m_seed / kk
    w_trail = None if spec.win_min is None else int(spec.win_min * 60 * SFREQ)
    # ewma state, seeded on the calibration block
    e_n, e_s1, e_s2 = kk, c1[k], c2[k]
    alpha_per_sample = (None if spec.tau_min is None
                        else np.exp(-1.0 / (spec.tau_min * 60.0 * SFREQ)))

    a = k
    while a < n:
        b = min(a + EPOCH, n)
        if spec.mode == "expand":
            nn = m_seed + (a - k)
            s1 = c1[k] * seed_r + (c1[a] - c1[k])
            s2 = c2[k] * seed_r + (c2[a] - c2[k])
        elif spec.mode == "ewma":
            nn, s1, s2 = e_n, e_s1, e_s2
        else:
            lo = max(a - w_trail, 0)
            hi = max(a, k)
            nn = float(hi - lo)
            s1, s2 = c1[hi] - c1[lo], c2[hi] - c2[lo]
        mu = s1 / nn
        sd = max(float(np.sqrt(max(s2 / nn - mu ** 2, TINY))), TINY)
        blk = np.clip(sig[a:b], mu - c * sd, mu + c * sd)
        xc[a:b] = blk
        c1[a + 1:b + 1] = c1[a] + np.cumsum(blk)
        c2[a + 1:b + 1] = c2[a] + np.cumsum(blk ** 2)
        if spec.mode == "ewma":
            d = alpha_per_sample ** (b - a)
            e_n = e_n * d + (b - a)
            e_s1 = e_s1 * d + blk.sum()
            e_s2 = e_s2 * d + (blk ** 2).sum()
        a = b
    return xc


# ======================================================================
# the modes
# ======================================================================
def _expanding_pipeline(sig, k, scale):
    """`pipeline`'s expanding z-score, copied as is so `expand20` is bit-identical."""
    n = len(sig)
    starts = np.arange(0, n - 4 * EPOCH, EPOCH)
    ends = np.maximum(starts + EPOCH, k)
    c1 = np.concatenate([[0.0], np.cumsum(sig)])
    c2 = np.concatenate([[0.0], np.cumsum(sig ** 2)])
    mu = c1[ends] / ends
    var = np.maximum(c2[ends] / ends - mu ** 2, 1e-30)
    sd = np.sqrt(var) * scale
    eps = np.stack([(sig[s:s + EPOCH] - m) / d
                    for s, m, d in zip(starts, mu, sd)])
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


def _expanding_weighted(sig, acc, k, w_min, scale, starts, ends):
    """Expanding, with the seed counted as `w_min` minutes.

    `sig` is z-scored, `acc` feeds the statistics; they differ only under `wins`.
    """
    c1, c2, c3, c4 = _cums(acc)
    m_seed = max(w_min * 60.0 * SFREQ, 1.0)
    r = m_seed / float(k)
    n_eff = m_seed + (ends - k).astype(float)
    s1 = c1[k] * r + (c1[ends] - c1[k])
    s2 = c2[k] * r + (c2[ends] - c2[k])
    s3 = c3[k] * r + (c3[ends] - c3[k])
    s4 = c4[k] * r + (c4[ends] - c4[k])
    pmax = np.maximum.accumulate(acc)[ends - 1]
    pmin = np.minimum.accumulate(acc)[ends - 1]
    mu, sd, chan = _from_raw_moments(s1, s2, s3, s4, n_eff, scale, pmax, pmin)
    return _cut(sig, starts, mu, sd), chan


def _ewma(sig, acc, k, tau_min, scale, starts, ends):
    """Exponentially weighted moments, decayed once per epoch.

    Starts as the calibration block's own estimate, which then decays like
    everything else.
    """
    alpha_s = np.exp(-1.0 / (tau_min * 60.0 * SFREQ))
    c1, c2, c3, c4 = _cums(acc)
    n_ep = len(starts)
    S = np.empty((n_ep, 4))
    N = np.empty(n_ep)
    st = np.array([c1[k], c2[k], c3[k], c4[k]], dtype=float)
    nn = float(k)
    prev = k
    for j, e in enumerate(ends):
        if e > prev:
            d = alpha_s ** (e - prev)
            st = st * d + np.array([c1[e] - c1[prev], c2[e] - c2[prev],
                                    c3[e] - c3[prev], c4[e] - c4[prev]])
            nn = nn * d + (e - prev)
            prev = e
        S[j] = st
        N[j] = nn
    pmax = np.maximum.accumulate(acc)[ends - 1]
    pmin = np.minimum.accumulate(acc)[ends - 1]
    mu, sd, chan = _from_raw_moments(S[:, 0], S[:, 1], S[:, 2], S[:, 3], N,
                                     scale, pmax, pmin)
    return _cut(sig, starts, mu, sd), chan


def _trailing(sig, acc, k, win_min, scale, starts, ends):
    """`pipeline`'s trailing z-score, with the window length as a parameter."""
    w = int(win_min * 60 * SFREQ)
    lo = np.maximum(ends - w, 0)
    n_eff = (ends - lo).astype(float)
    c1, c2, c3, c4 = _cums(acc)
    pmax = np.array([acc[a:b].max() for a, b in zip(lo, ends)])
    pmin = np.array([acc[a:b].min() for a, b in zip(lo, ends)])
    mu, sd, chan = _from_raw_moments(c1[ends] - c1[lo], c2[ends] - c2[lo],
                                     c3[ends] - c3[lo], c4[ends] - c4[lo],
                                     n_eff, scale, pmax, pmin)
    return _cut(sig, starts, mu, sd), chan


# ======================================================================
# the entry point
# ======================================================================
def build_epochs(sig, spec):
    """`pipeline.build_epochs` for any spec above."""
    spec = parse(spec) if not isinstance(spec, NormSpec) else spec
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    k = int(spec.calib_sec * SFREQ)
    if k <= 0 or k >= n:
        raise ValueError(f"calibration block ({k} samples) does not fit in "
                         f"{n} samples of signal")
    starts, ends = _epoch_grid(n, k)

    # expand20: the shipped path, bit-identical to pipeline
    if spec.mode == "expand" and spec.seed_w is None and spec.wins_c is None:
        return _expanding_pipeline(sig, k, spec.scale)

    acc = sig if spec.wins_c is None else _winsorise(sig, spec, starts, ends, k)

    if spec.mode == "expand":
        w = spec.seed_w if spec.seed_w is not None else spec.calib_sec / 60.0
        return _expanding_weighted(sig, acc, k, w, spec.scale, starts, ends)
    if spec.mode == "ewma":
        return _ewma(sig, acc, k, spec.tau_min, spec.scale, starts, ends)
    return _trailing(sig, acc, k, spec.win_min, spec.scale, starts, ends)


# ======================================================================
# self-test: expand20 == pipeline's expanding, bit for bit
# ======================================================================
def check_expand20(sig, calib_sec=None):
    """True if `expand20` matches `pipeline.build_epochs` exactly."""
    cs = S.CALIB_SEC if calib_sec is None else calib_sec
    a_e, a_c = build_epochs(sig, NormSpec(mode="expand", calib_sec=cs))
    b_e, b_c = P.build_epochs(sig, P.Normalizer("expanding", calib_sec=cs))
    ok = np.array_equal(a_e, b_e)
    for key in ("chan_ptp", "chan_skew", "chan_kurt"):
        ok &= np.array_equal(np.asarray(a_c[key]), np.asarray(b_c[key]))
    return bool(ok), (a_e, a_c), (b_e, b_c)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400_000) * 3e-5
    x[::5000] *= 12.0
    ok, _, _ = check_expand20(x, calib_sec=120.0)
    print("expand20 == pipeline expanding (synthetic):", ok)
    for nm in ("expand20", "expandw5", "expandw40", "ewma20", "trail20",
               "expandw5+wins6"):
        sp = parse(nm)
        sp._calib_sec = 120.0
        e, c = build_epochs(x, sp)
        print(f"  {nm:<16s} -> {sp!r} epochs {e.shape} "
              f"ptp[-1] {np.atleast_1d(c['chan_ptp'])[-1]:.3f}")
