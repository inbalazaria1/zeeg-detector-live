#!/usr/bin/env python
"""
Read a few channels out of an EGI .mff without MNE.

MNE walks every block header and reads all 257 channels, which was far too
slow on the Google Drive mount the recordings lived on.  Here only the first
block has a header; every later block starts with a 4-byte 0, so block k's
offset can be computed, and each channel inside a block is one contiguous
run.  Two channels are then two small reads per block.

Scaling is MNE's: 1e-6 * GCAL gain per channel.  `--verify-dh3` compared this
against MNE on DH3 and matched exactly (it needs a cache not included here).

From dev/research:

    ../../.venv/bin/python _ext/mf7/mff_direct.py --info "_ext/mff's/XY1.mff"
"""
import argparse
import os
import re
import struct
import sys
import time

import numpy as np


def read_gcal(info_xml):
    """The GCAL per-channel gain table out of infoN.xml, 1-based order."""
    with open(info_xml) as fh:
        txt = fh.read()
    blocks = re.findall(r"<calibration>(.*?)</calibration>", txt, re.S)
    for b in blocks:
        if "<type>GCAL</type>" not in b:
            continue
        pairs = re.findall(r'<ch n="(\d+)">([^<]+)</ch>', b)
        g = np.empty(len(pairs))
        for n, v in pairs:
            g[int(n) - 1] = float(v)
        return g
    return None


class MffSignal:
    """Block geometry of one signalN.bin, assuming a single uniform epoch."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(16)
            version, self.header_size, self.block_size, self.nc = struct.unpack(
                "<4i", head)
            if version != 1:
                raise ValueError(f"first block is version {version}, expected 1")
            rest = fh.read(self.header_size - 16)
        self.offsets = np.frombuffer(rest, dtype="<i4", count=self.nc)
        sigfreq = np.frombuffer(rest, dtype="<i4", count=self.nc,
                                offset=4 * self.nc)
        depth = sigfreq[0] & 0xFF
        if depth != 32:
            raise ValueError(f"depth {depth} != 32")
        self.sfreq = float(sigfreq[0] >> 8)
        self.samples_per_block = self.block_size // 4 // self.nc

        # block 0 is header + data; every later block is a 4-byte 0 + data
        self.stride = 4 + self.block_size
        self.first_end = self.header_size + self.block_size
        # Whole blocks only.  check_markers() catches a wrong stride, since a
        # misaligned offset almost never lands on a 0.
        self.n_blocks = 1 + (self.size - self.first_end) // self.stride
        self.tail_bytes = self.size - (self.first_end
                                       + (self.n_blocks - 1) * self.stride)
        if self.tail_bytes:
            # the last block is usually cut short when recording stops, and
            # may have its own header: drop it, it is the last 4 s
            self.n_blocks -= 1
        self.n_times = self.n_blocks * self.samples_per_block

    def data_start(self, k):
        """Byte offset of block k's data, past its header or marker."""
        return self.header_size if k == 0 else self.first_end + (k - 1) * self.stride + 4

    def check_markers(self, ks):
        """Raise unless each block in `ks` starts with the 0 marker."""
        with open(self.path, "rb") as fh:
            for k in ks:
                if k == 0:
                    continue
                fh.seek(self.first_end + (k - 1) * self.stride)
                v = struct.unpack("<i", fh.read(4))[0]
                if v != 0:
                    raise ValueError(f"block {k} marker is {v}, expected 0")

    def read(self, chan_idx, start, stop, progress=None):
        """Samples [start, stop) of the given 0-based channel indices, raw units."""
        stop = min(stop, self.n_times)
        s = self.samples_per_block
        k0, k1 = start // s, (stop - 1) // s + 1
        # spot-check the computed offsets before trusting them
        self.check_markers(sorted({k0, k1 - 1, *range(k0, k1, 32)}))
        out = np.empty((len(chan_idx), stop - start), dtype=np.float32)
        t0 = time.time()
        with open(self.path, "rb") as fh:
            for k in range(k0, k1):
                base = self.data_start(k)
                b0 = k * s                       # first sample of this block
                lo = max(start, b0) - b0         # slice wanted inside the block
                hi = min(stop, b0 + s) - b0
                for j, ci in enumerate(chan_idx):
                    fh.seek(base + int(self.offsets[ci]) + 4 * lo)
                    buf = fh.read(4 * (hi - lo))
                    if len(buf) != 4 * (hi - lo):
                        raise IOError(f"short read in block {k}, channel {ci}")
                    a = b0 + lo - start
                    out[j, a:a + hi - lo] = np.frombuffer(buf, dtype="<f4")
                if progress and (k - k0) % progress == 0:
                    done, tot = k - k0 + 1, k1 - k0
                    el = time.time() - t0
                    print(f"  block {done}/{tot}  {el:6.0f}s elapsed, "
                          f"eta {el / done * (tot - done):6.0f}s", flush=True)
        return out


def channel_index(mff, names):
    """0-based index into signal1.bin for EGI names E1..E256 (VREF/Cz is 257)."""
    idx = []
    for n in names:
        if n.upper() in ("VREF", "CZ"):
            idx.append(256)
        else:
            m = re.fullmatch(r"[Ee](\d+)", n)
            if not m:
                raise ValueError(f"cannot map channel name {n!r}")
            idx.append(int(m.group(1)) - 1)
    return idx


def read_channels(mff, names, tmin_sec, tmax_sec, progress=None):
    """(n_ch, n) in volts, the same as MNE's read_raw_egi().get_data()."""
    sig = MffSignal(os.path.join(mff, "signal1.bin"))
    gcal = read_gcal(os.path.join(mff, "info1.xml"))
    idx = channel_index(mff, names)
    start = int(round(tmin_sec * sig.sfreq))
    stop = int(round(tmax_sec * sig.sfreq))
    raw = sig.read(idx, start, stop, progress=progress)
    cal = np.array([1e-6 * (gcal[i] if gcal is not None else 1.0) for i in idx])
    return raw.astype(np.float64) * cal[:, None], sig


def _verify_dh3():
    """Compare with a DH3 cache built through MNE (not included here)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mff = os.path.join(root, "mff's", "DH3_20220529_120959.mff")
    npz = os.path.join(root, "simulator-without-cleaning", "rt", "cache",
                       "DH3_raw_22_75.npz")
    z = np.load(npz, allow_pickle=True)
    ref = z["data"]
    print(f"reference {ref.shape} from {os.path.basename(npz)} "
          f"(crop {z['crop_min']} min, {list(z['ch_names'])})")

    sig = MffSignal(os.path.join(mff, "signal1.bin"))
    print(f"DH3 signal1.bin: {sig.n_blocks} blocks x {sig.samples_per_block} "
          f"samples x {sig.nc} ch @ {sig.sfreq} Hz = {sig.n_times / 60000:.1f} min")

    # 10 s is enough: a wrong channel index or gain shows up at once
    t0 = 22.0 * 60
    got, _ = read_channels(mff, ["E227", "E254"], t0, t0 + 10)
    exp = ref[:, :10 * 1000]
    err = np.abs(got - exp).max()
    print(f"max |direct - MNE| over 10 s x 2 ch : {err:.3e} V "
          f"(signal std {exp.std():.3e} V)")
    ok = err < 1e-15
    print("VERIFY:", "identical" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dh3", action="store_true")
    ap.add_argument("--info", metavar="MFF")
    a = ap.parse_args()
    if a.verify_dh3:
        sys.exit(_verify_dh3())
    if a.info:
        s = MffSignal(os.path.join(a.info, "signal1.bin"))
        print(f"{s.n_blocks} blocks, {s.samples_per_block} samples/block, "
              f"{s.nc} channels, {s.sfreq} Hz, {s.n_times} samples "
              f"= {s.n_times / s.sfreq / 60:.2f} min")
