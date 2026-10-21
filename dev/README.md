# `dev/` — not needed to run the detector

**Nothing in this directory is required to run anything.** `../detect.ipynb`
needs `../models/`, `../constants.py` and `../requirements.txt`, and nothing in
here. You can delete this folder and the detector still works.

It exists so the correction can be refitted, and so the choices baked into the
detector can be checked instead of taken on trust.

| file | what it is | when you need it |
|------|------------|------------------|
| [`train.ipynb`](train.ipynb) | the recipe for model 2, and the code that fits it | you have new donor recordings and want to refit the correction |
| [`research/`](research/README.md) | the code that turns a recording into training matrices, from the lab's research package | you are adding a subject |
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

## Adding a new subject

A subject is in the training set if `load_training_windows()` returns it.
Getting it there means building its matrices with the code in
[`research/`](research/README.md), then adding it to the loader.

The shipped model was fitted on three recordings, 13 windows, 584 min:

| subject | windows |
|---------|---------|
| DH3 | c1, c2, c4 |
| AO0 | c3, c4, c5, c6 |
| MF7 | c1, c2, c3, c4, c5, c6 |

The same list is in `../models/correction_meta.json` under `windows`, and is
rewritten on every refit.

Paths below are inside `dev/research/`. `XY1` stands for the new subject.

1. **Put the data in.** None of it is committed, `.gitignore` keeps it out.
   * the recording: `_ext/mff's/XY1.mff` (EGI, 1000 Hz, E227 and E254);
   * the hypnogram: `_ext/scoring-sleep_stages/XY1.txt`, one stage per second,
     whitespace separated: 0 W, 1 N1, 2 N2, 3 N3, 4 REM, -1 unscored. The
     shipped ones are expert scorings in this format. It has to cover the
     whole recording, because NREM periods past the end of the scoring are
     ignored. If yours has one stage per 30 s epoch, set `epoch_sec=30` in
     step 2 instead of 1.
2. **Register it** in `SUBJECTS` in `_ext/cycles/hypno.py`:

   ```python
   "XY1": dict(rec="XY1", mff="mff's/XY1.mff",
               hypno=os.path.join(SCORING, "XY1.txt"), epoch_sec=1,
               source="expert", rec_min=400.0, max_min=400.0),
   ```

   `rec_min` is the recording length in minutes. `max_min` is how far it is
   safe to read, normally the same. It is less only if the `.mff` has a gap
   (more than one epoch in `epochs.xml`, inside the `.mff` folder), see the
   DH3 note in `hypno.py`. The reader cannot handle a gap, so for a recording
   with one, set `max_min` to the end of the first epoch.
   Then add `"XY1"` to `EXTRA_DONORS` in `sleep/windows.py`.
3. **Check the windows and build.** Every NREM period of at least 25 min
   becomes a window (`MIN_TRAINING_WINDOW_MINUTES` in `../constants.py`).
   Shorter ones are no use: each window's first 20 min is the normalisation
   seed and gets dropped, so there is almost nothing left to train on.

   ```bash
   cd dev/research
   ../../.venv/bin/python sleep/build.py --plan --subjects XY1 --kinds causal,offline
   ../../.venv/bin/python sleep/build.py --all  --subjects XY1 --kinds causal,offline
   ```

   `--plan` lists the windows and builds nothing. `--all` reads the night out
   of the `.mff` once, then builds two matrices per window, about 40 s each:
   `causal`, the 25 ms chain the detector runs, and `offline`, the 16.5 s
   lookahead chain that gives the teacher score. They land in `cache/sleep/`.
4. **Get the old donors' matrices too.** The fit uses every window, not just
   the new ones, so you cannot skip this. Build `{DH3,AO0,MF7}` the same way
   from their `.mff` and hypnograms, or put matrices you already have in
   `cache/sleep/`. Only the code was copied here, no matrices or recordings,
   so the refit cannot run until one of the two is done.
5. **Point the loader at it.** Replace `load_training_windows()` in
   `train.ipynb` with:

   ```python
   R = os.path.join(REPO_ROOT, "dev", "research")
   sys.path[:0] = [R, os.path.join(R, "sleep")]
   import build as B, common as C, windows as W

   def load_training_windows():
       out = {}
       for subject in ("DH3", "AO0", "MF7", "XY1"):
           out[subject] = []
           for tag, lo, hi, _ in W.long_windows(subject):
               xc = B.get(subject, tag, lo, hi, "causal", build=False)
               xo = B.get(subject, tag, lo, hi, "offline", build=False)
               out[subject].append(make_training_window(
                   f"{subject}/{tag}", xc, C.proba(xc), C.proba(xo)))
       return out
   ```

   With only the first three subjects this refits the shipped
   `correction.json` byte for byte, which is a good first run to make sure
   the setup is right before adding `XY1`.
6. **Run All**, then go through the notebook's "After refitting" cell.

Things to watch:

* The subject list in the loader is the training set. `windows.py` also lists
  ON68, the healthy control, but the shipped model was not trained on it.
* Selection holds out one key at a time. If a patient has two nights, put both
  under one key, otherwise the held-out fold has seen that patient.
* A subject is only held out if at least one of its windows has 20 or more
  scored reference spikes (`MIN_VALIDATION_REF_SPIKES`). If none do it still
  trains, but it never gets validated.
* `LOSO_PRECISION / RECALL / F1` in the final-fit cell are the shipped model's
  numbers (0.388, 0.392, 0.383). After adding a donor they no longer describe
  what you saved, and this notebook does not measure new ones. Either replace
  them by hand with the pooled numbers from the selection step, or delete the
  `loso_estimate` entry from `correction_meta.json`, so the file does not
  quote numbers for a model they were not measured on.
* Adding a donor adds a leave-one-out fold, so the chosen depth and alpha can
  change. Both are written into `correction_meta.json` and `../detect.ipynb`
  picks them up at load.

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

The correction-on acceptance test (267 detections) and the synthetic
fingerprint depend on `correction.json`, so expect them to change. Check that
the new values are sensible, then update them in `../detect.ipynb` and in the
"Verifying" section of `../README.md`. The README's published numbers are for
the shipped model only.

## What it cannot do here

`train.ipynb` needs per-window design matrices built from the raw `.mff`
recordings: ~570 MB, derived from ~52 GB of patient data that is not
distributed. It documents the data contract it needs, so you can point it at
your own recordings, but **it will not run from a clone of this repository**.
The notebook says so in its first cell and raises instead of pretending.
