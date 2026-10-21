#!/usr/bin/env python
"""
Stage-6 normalisation variants, a superset of `pipeline.build_epochs`.

An extended copy of `norms.py`.  Everything above the "chan_* overrides"
section is unchanged, so `expand20` is still bit-identical to
`pipeline.Normalizer("expanding")`.  `build.py` imports this module only for
`expandg10`.

Why the chan_* overrides exist: six of the 210 columns, `chan_ptp`, `chan_skew`
and `chan_kurt` per channel, carry about a third of the teacher's stage-6 loss.
The booster was trained with them as one constant per channel for the whole
night.  Under `expand20` they are running values, and skew and kurtosis are
dominated by the few largest samples seen so far, so one artefact in the
calibration block skews them for hours (DH3: `chan_kurt` 78.6 at minute 20
against a whole-crop 9.6).  `chan_ptp` is a running max minus a running min and
never comes back down.  The running sd is fine: it drifts smoothly to the
whole-crop value.  All measured in the research tree.

Contract, same as `pipeline.build_epochs`:

    build_epochs(sig, spec) -> (epochs, {"chan_ptp":.., "chan_skew":.., "chan_kurt":..})

`epochs` is (n_epochs, 250) of z-scored signal; each `chan_*` is per epoch, from
the same statistics as the z-score.  Except `global`, nothing reads past the end
of the epoch being decided.

Spec names: one base mode, then modifiers, joined by `+`.

  base modes
    expand20      expanding, seeded on the 20 min calibration block.  The
                  baseline (`expand` is a synonym).
    expandw{W}    expanding, with the seed counted as W minutes of samples
                  instead of 20.  W < 20 lets the live signal take over sooner.
                  Numerically, not bit-wise, equal to expand20 at W = 20.
    ewma{T}       exponentially weighted moments, time constant T minutes,
                  seeded on the calibration block.
    trail{W}      trailing W minutes, never shorter than the calibration block.
    expandg{F}    expanding from the first sample, the window floored at F
                  seconds instead of 20 min.  `expand20` gives every epoch in
                  the seed the whole block's statistics; this normalises the
                  seed like the rest of the night.  No scored epoch changes, only
                  the first 20 min.  `expandg1200` is `expand20`.
    global        not causal: the whole crop's mean, sd and chan_*, exactly
                  `pipeline.Normalizer("global")`.  A ceiling, never deployable.

  modifiers
    wins{c}       clip samples to +-c sd before they enter the accumulator.
    sc{c}         multiply the z-score divisor by a constant c (and divide
                  chan_ptp by it).  One number for everyone, set in advance.

  chan_* overrides: change only the six channel columns, so a variant can be
  screened by patching a cached matrix.  At most one of:
    cfreeze       hold chan_* at their value at the end of the calibration
                  block, all night.  Unlike `calib`, the z-score keeps running.
    cwins{c}      chan_* from a stream clipped to +-c running sd; the z-score
                  uses the unclipped one.
    cconst        three constants from outside the night
                  (`NormSpec(chan_const=...)`, no name syntax).

Choices worth knowing:
* ewma decays once per 250 ms epoch, not per sample, the way a device would
  update.  The factor is `exp(-len(block)/SFREQ / (T*60))`, so T minutes of
  memory means T minutes.
* Clipping uses the running estimate through the previous epoch, not a band
  frozen on the calibration block: the detrended amplitude nearly triples
  across a NREM period on DH3, so a fixed band would clip more real signal as
  the night goes on.  The seed is clipped against its own mean/sd.  Only the
  accumulator sees clipped data; the z-scored epoch is never clipped.
* Under `wins`, chan_ptp's extremes come from the clipped stream too.
* chan_ptp always divides by the z-score's sd, so it stays on the same scale
  as the epochs next to it.
* `cfreeze` reads index 0: `_epoch_grid` clamps every end to at least k, so
  epoch 0 already holds the full calibration block.
"""
import os
import sys

import numpy as np
import scipy.stats as sp_stats

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
ROOT = os.path.join(TRACK, "_ext")     # the copied research modules
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
    """One stage-6 choice.  `parse()` builds it from a name.

    mode         'expand' | 'expandg' | 'ewma' | 'trail' | 'global'
    seed_w       expand: seed weight in minutes, None = true length
    tau_min      ewma: time constant, minutes
    win_min      trail: window, minutes
    floor_sec    expandg: window floor, seconds
    wins_c       clip half-width in sd, or None
    scale        multiplier on the z-score divisor, 1.0 = none
    calib_sec    calibration length, default `subjects.CALIB_SEC`
    chan_freeze  hold chan_* at the calibration block's value
    chan_wins    clip half-width for the chan_* stream, or None
    chan_const   {"chan_ptp": v, "chan_skew": v, "chan_kurt": v}, or None
    """

    CHAN_KEYS = ("chan_ptp", "chan_skew", "chan_kurt")

    def __init__(self, mode="expand", seed_w=None, tau_min=None, win_min=None,
                 wins_c=None, calib_sec=None, scale=1.0,
                 chan_freeze=False, chan_wins=None, chan_const=None,
                 floor_sec=None):
        if mode not in ("expand", "expandg", "ewma", "trail", "global"):
            raise ValueError(f"unknown norm mode {mode!r}")
        if mode == "expandg" and not floor_sec:
            raise ValueError("expandg needs a floor, e.g. expandg10")
        if mode == "ewma" and not tau_min:
            raise ValueError("ewma needs a time constant, e.g. ewma30")
        if mode == "trail" and not win_min:
            raise ValueError("trail needs a window, e.g. trail20")
        if float(scale) <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        n_chan = sum([bool(chan_freeze), chan_wins is not None,
                      chan_const is not None])
        if n_chan > 1:
            raise ValueError("cfreeze / cwins / cconst are mutually exclusive; "
                             "each is a complete answer to where chan_* comes "
                             "from")
        if chan_const is not None:
            missing = [k for k in self.CHAN_KEYS if k not in chan_const]
            if missing:
                raise ValueError(f"chan_const is missing {missing}")
        self.mode = mode
        self.floor_sec = None if floor_sec is None else float(floor_sec)
        self.seed_w = None if seed_w is None else float(seed_w)
        self.tau_min = None if tau_min is None else float(tau_min)
        self.win_min = None if win_min is None else float(win_min)
        self.wins_c = None if wins_c is None else float(wins_c)
        self._calib_sec = calib_sec
        self.scale = float(scale)
        self.chan_freeze = bool(chan_freeze)
        self.chan_wins = None if chan_wins is None else float(chan_wins)
        self.chan_const = (None if chan_const is None
                           else {k: float(chan_const[k]) for k in self.CHAN_KEYS})

    @property
    def calib_sec(self):
        return S.CALIB_SEC if self._calib_sec is None else self._calib_sec

    def name(self):
        if self.mode == "global":
            base = "global"
        elif self.mode == "expandg":
            base = f"expandg{self.floor_sec:g}"
        elif self.mode == "expand":
            base = "expand20" if self.seed_w is None else f"expandw{self.seed_w:g}"
        elif self.mode == "ewma":
            base = f"ewma{self.tau_min:g}"
        else:
            base = f"trail{self.win_min:g}"
        if self.wins_c is not None:
            base += f"+wins{self.wins_c:g}"
        if abs(self.scale - 1.0) > 1e-12:
            base += f"+sc{self.scale:g}"
        if self.chan_freeze:
            base += "+cfreeze"
        if self.chan_wins is not None:
            base += f"+cwins{self.chan_wins:g}"
        if self.chan_const is not None:
            # the caller says where the numbers came from
            base += "+cconst"
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
    if base == "global":
        kw.update(mode="global")
    elif base.startswith("expandg"):
        kw.update(mode="expandg", floor_sec=float(base[len("expandg"):]))
    elif base in ("expand20", "expand", "expanding"):
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
        elif m == "cfreeze":
            kw["chan_freeze"] = True
        elif m.startswith("cwins"):
            kw["chan_wins"] = float(m[len("cwins"):])
        else:
            raise ValueError(f"unknown norm modifier {m!r}")
    return NormSpec(**kw)


# ======================================================================
# helpers
# ======================================================================
def _epoch_grid(n, k):
    """`pipeline.build_epochs`'s epoch starts and the ends its statistics use."""
    starts = np.arange(0, n - 4 * EPOCH, EPOCH)
    ends = np.maximum(starts + EPOCH, k)
    return starts, ends


def _cums(x):
    """Prefix sums of x, x^2, x^3, x^4 (length n+1, c[0] = 0)."""
    return [np.concatenate([[0.0], np.cumsum(x ** p)]) for p in (1, 2, 3, 4)]


def _from_raw_moments(s1, s2, s3, s4, n, scale, pmax, pmin):
    """(mu, sd, chan_*) from raw power SUMS and an effective sample count."""
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
def _winsorise(sig, spec, starts, ends, k, c=None):
    """The clipped stream, same length as `sig`.

    Sequential because the band is the running estimate: the calibration block
    is clipped against its own mean/sd, each later epoch against the estimate
    through the previous epoch.  `c` defaults to `spec.wins_c`; `cwins` passes
    its own, so both modifiers share one clipping rule.
    """
    c = spec.wins_c if c is None else float(c)
    xc = np.empty_like(sig, dtype=float)
    mu0, sd0 = float(np.nanmean(sig[:k])), float(np.nanstd(sig[:k]))
    sd0 = max(sd0, TINY)
    xc[:k] = np.clip(sig[:k], mu0 - c * sd0, mu0 + c * sd0)

    n = len(sig)
    # prefix sums of the clipped stream, filled as we go (ewma has its own)
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
    """`pipeline.build_epochs(..., Normalizer('expanding'))`, copied as is.

    A literal copy, so `expand20` is bit-identical by construction.
    """
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
    """Expanding, with the seed weighted as `w_min` minutes.

    `sig` is z-scored, `acc` feeds the accumulator; they differ only under `wins`.
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
    """Exponentially weighted raw moments, decayed once per epoch.

    Seeded with the calibration block's sums and true count, so when scoring
    starts the estimate equals the calibration one, then the seed decays.
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
    """`pipeline`'s `trailing`, with the window length as a free parameter."""
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
# chan_* overrides
# ======================================================================
def _mode_sums(acc, spec, k, ends):
    """`(n_eff, [s1..s4], pmax, pmin)` for `acc` under the spec's base mode.

    Written once so the chan_* override and the z-score accumulate the same way.
    """
    c1, c2, c3, c4 = _cums(acc)
    if spec.mode == "expand":
        w_min = spec.seed_w if spec.seed_w is not None else spec.calib_sec / 60.0
        m_seed = max(w_min * 60.0 * SFREQ, 1.0)
        r = m_seed / float(k)
        n = m_seed + (ends - k).astype(float)
        s = [c[k] * r + (c[ends] - c[k]) for c in (c1, c2, c3, c4)]
        pmax = np.maximum.accumulate(acc)[ends - 1]
        pmin = np.minimum.accumulate(acc)[ends - 1]
    elif spec.mode == "ewma":
        alpha_s = np.exp(-1.0 / (spec.tau_min * 60.0 * SFREQ))
        n_ep = len(ends)
        S = np.empty((n_ep, 4))
        N = np.empty(n_ep)
        st = np.array([c1[k], c2[k], c3[k], c4[k]], dtype=float)
        nn, prev = float(k), k
        for j, e in enumerate(ends):
            if e > prev:
                d = alpha_s ** (e - prev)
                st = st * d + np.array([c1[e] - c1[prev], c2[e] - c2[prev],
                                        c3[e] - c3[prev], c4[e] - c4[prev]])
                nn = nn * d + (e - prev)
                prev = e
            S[j], N[j] = st, nn
        n = N
        s = [S[:, 0], S[:, 1], S[:, 2], S[:, 3]]
        pmax = np.maximum.accumulate(acc)[ends - 1]
        pmin = np.minimum.accumulate(acc)[ends - 1]
    else:
        w = int(spec.win_min * 60 * SFREQ)
        lo = np.maximum(ends - w, 0)
        n = (ends - lo).astype(float)
        s = [c[ends] - c[lo] for c in (c1, c2, c3, c4)]
        pmax = np.array([acc[a:b].max() for a, b in zip(lo, ends)])
        pmin = np.array([acc[a:b].min() for a, b in zip(lo, ends)])
    return n, s, pmax, pmin


def chan_override(sig, spec, starts, ends, k, chan):
    """`chan` with the spec's chan_* override applied, as a new dict.

    `sig` is the unclipped signal the z-score used, so `sd_z` here is the
    epochs' own divisor.
    """
    if spec.chan_const is not None:
        m = len(starts)
        return {kk: np.full(m, spec.chan_const[kk], dtype=float)
                for kk in NormSpec.CHAN_KEYS}
    if spec.chan_freeze:
        # epoch 0's end is clamped to k, so it holds the calibration block
        return {kk: np.full(len(starts), float(np.atleast_1d(chan[kk])[0]))
                for kk in NormSpec.CHAN_KEYS}
    if spec.chan_wins is None:
        return dict(chan)

    acc = _winsorise(sig, spec, starts, ends, k, c=spec.chan_wins)
    n_c, s_c, pmax, pmin = _mode_sums(acc, spec, k, ends)
    # skew / kurt from the clipped accumulator: they are scale-free
    _, _, ch = _from_raw_moments(s_c[0], s_c[1], s_c[2], s_c[3], n_c,
                                 spec.scale, pmax, pmin)
    # ptp keeps the z-score's divisor, to match the epochs
    base_acc = sig if spec.wins_c is None else _winsorise(sig, spec, starts,
                                                          ends, k)
    n_z, s_z, _, _ = _mode_sums(base_acc, spec, k, ends)
    _, sd_z, _ = _from_raw_moments(s_z[0], s_z[1], s_z[2], s_z[3], n_z,
                                   spec.scale, pmax, pmin)
    ch["chan_ptp"] = (pmax - pmin) / sd_z
    return ch


def has_chan_override(spec):
    spec = parse(spec)
    return (spec.chan_freeze or spec.chan_wins is not None
            or spec.chan_const is not None)


# ======================================================================
# the entry point
# ======================================================================
def build_epochs(sig, spec):
    """Same contract as `pipeline.build_epochs`, for the specs above."""
    spec = parse(spec) if not isinstance(spec, NormSpec) else spec
    sig = np.asarray(sig, dtype=float)
    n = len(sig)

    if spec.mode == "global":
        # not causal.  Delegated to `pipeline` so it is the same arithmetic
        # the offline chain uses.
        return P.build_epochs(sig, P.Normalizer("global", scale=spec.scale))

    k = int(spec.calib_sec * SFREQ)
    if k <= 0 or k >= n:
        raise ValueError(f"calibration block ({k} samples) does not fit in "
                         f"{n} samples of signal")
    if spec.mode == "expandg":
        # expand20's code with the floor at `floor_sec`; expandg1200 = expand20
        kf = max(int(spec.floor_sec * SFREQ), EPOCH)
        starts, ends = _epoch_grid(n, kf)
        eps, chan = _expanding_pipeline(sig, kf, spec.scale)
        return eps, chan_override(sig, spec, starts, ends, kf, chan)

    starts, ends = _epoch_grid(n, k)

    # the bit-identical path, kept even with a chan_* override: going through
    # `_expanding_weighted` would sum seed + increment instead of one cumsum and
    # change the epochs in the last bits
    if spec.mode == "expand" and spec.seed_w is None and spec.wins_c is None:
        eps, chan = _expanding_pipeline(sig, k, spec.scale)
        return eps, chan_override(sig, spec, starts, ends, k, chan)

    acc = sig if spec.wins_c is None else _winsorise(sig, spec, starts, ends, k)

    if spec.mode == "expand":
        w = spec.seed_w if spec.seed_w is not None else spec.calib_sec / 60.0
        eps, chan = _expanding_weighted(sig, acc, k, w, spec.scale, starts, ends)
    elif spec.mode == "ewma":
        eps, chan = _ewma(sig, acc, k, spec.tau_min, spec.scale, starts, ends)
    else:
        eps, chan = _trailing(sig, acc, k, spec.win_min, spec.scale, starts, ends)
    return eps, chan_override(sig, spec, starts, ends, k, chan)


# ======================================================================
# self-test: expand20 == pipeline's expanding, bit for bit
# ======================================================================
def check_expand20(sig, calib_sec=None):
    """Bit-identity of `expand20` against `pipeline.build_epochs`."""
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
               "expandw5+wins6", "expand20+cfreeze", "expand20+cwins4",
               "trail20+cwins4", "ewma20+cfreeze"):
        sp = parse(nm)
        sp._calib_sec = 120.0
        e, c = build_epochs(x, sp)
        print(f"  {nm:<18s} -> {sp!r} epochs {e.shape} "
              f"ptp[-1] {np.atleast_1d(c['chan_ptp'])[-1]:.3f} "
              f"kurt[-1] {np.atleast_1d(c['chan_kurt'])[-1]:.3f}")
    sp = NormSpec(mode="expand", calib_sec=120.0,
                  chan_const=dict(chan_ptp=20.0, chan_skew=0.5, chan_kurt=9.0))
    e, c = build_epochs(x, sp)
    print(f"  {sp.name():<18s} -> chan_ptp {c['chan_ptp'][:2]} "
          f"chan_kurt {c['chan_kurt'][:2]}")
