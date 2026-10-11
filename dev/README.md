# `dev/` — not needed to run the detector

**Nothing in this directory is required to run anything.** `../detect.ipynb`
needs `../models/`, `../constants.py` and `../requirements.txt`, and nothing in
here. You can delete this folder and the detector still works.

It exists so the correction can be refitted, and so the choices baked into the
detector can be checked instead of taken on trust.

| file | what it is | when you need it |
|------|------------|------------------|
| [`train.ipynb`](train.ipynb) | the recipe for model 2, and the code that fits it | you have new donor recordings and want to refit the correction |
| [`RESULTS.md`](RESULTS.md) | the full write-up: what was tried, what was rejected, and the measured numbers | you want to know *why* a constant is what it is, or you are about to change one |

---

## Before you change anything

`RESULTS.md` is long, so here is what it will tell you if you go looking for the
easy wins. Each of these was tried and measured.

* **Moving the threshold off 0.70 loses.** Four rules were tried, including
  fitting it on the patient's own normalisation block. Mean precision is flat
  from 0.70 to 0.80 while recall collapses, and precision above ~0.39 is not
  reachable by thresholding at all (§2, §6).
* **Let the fit choose less.** Only two settings are searched at all — tree
  depth, and α, how much of the correction to add. The threshold is not one of
  them: it is fixed at 0.70. Every version that also let the fit choose the
  threshold scored worse, and got worse the more training data it had — the
  sign that it is fitting the choice, not the data (§2, §6, §11). Both are
  picked once, at training time, and frozen into `../models/correction.json`
  (depth 6, α 0.4). Nothing is chosen while the detector runs.
* **Swapping the stage-1 filters is not worth it.** An FIR high-pass, the other
  IIR types, and half the lookahead all measured the same or worse. The only
  knob that moves F1 is the Butterworth order, and the three patients want
  three different orders (2, 3, 4), so the shipped order 3 is a compromise, not
  a winner. This one is *not* in `RESULTS.md` — it was measured in a separate
  filter-variants experiment, which is not part of this repository.

## What `train.ipynb` does

```
1  windows   every NREM period >= 25 min of every donor recording, each one
              featurised twice: the causal 25 ms chain the detector runs, and
              the offline 16.5 s-lookahead chain that plays teacher
2  rows      drop each window's first 20 min (the z-score seed), stack the
              rest, add model 1's own logit as the last column
              target = logit(teacher) - logit(frozen)
3  select    depth and alpha, leave-one-recording-out at threshold 0.70,
              2 depths x 7 alphas, best mean pooled exact-epoch F1
4  fit       one booster on every window at the chosen depth, deterministic
5  save      ../models/correction.json and correction_meta.json
```

Step 1 is the expensive one and does not happen in this notebook — you bring
the matrices. Steps 2–5 take about a minute.

## Running `train.ipynb`

```bash
./.venv/bin/jupyter lab dev/train.ipynb    # from the repo root, same venv as detect.ipynb
```

Point `load_training_windows()` at your own cache, then Run All. Out of the box
it raises: no design matrices ship with this repository. What it needs is in the
notebook's "What you need before this will run" cell, and why is at the bottom
of this file.

## What refitting does and does not change

`train.ipynb` writes `../models/correction.json` and its metadata. It does
**not** touch:

* model 1, the frozen classifier, which is never refit and never re-saved;
* the signal chain, the 210 feature definitions, or stage 7;
* the threshold;
* `../constants.py`, which it imports rather than defines. Both notebooks read
  the same file, so a constant cannot mean one thing at training time and
  another at inference. If you do change one there, `../detect.ipynb` asserts
  the affected values against `../models/correction_meta.json` on load and fails
  rather than running with a mismatch.

After a refit, the frozen-only acceptance test in `../detect.ipynb` (291
detections) must still pass unchanged. If it moves, the problem is in the chain,
not in the correction.

## What it cannot do here

`train.ipynb` needs per-window design matrices built from the raw `.mff`
recordings: ~570 MB, derived from ~52 GB of patient data that is not
distributed. It documents the data contract it needs, so you can point it at
your own recordings, but **it will not run from a clone of this repository**.
The notebook says so in its first cell and raises instead of pretending.
