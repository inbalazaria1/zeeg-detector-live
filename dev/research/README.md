# `dev/research/`: recording to training matrices

The code that turns a recording into what `../train.ipynb` trains on: find the
NREM periods in the hypnogram, read E227 / E254 out of the `.mff`, and build
the `causal` and `offline` matrices for each period. How to use it is in
[`../README.md`](../README.md), "Adding a new subject".

It comes from `bin20-correction-nrem`, the research tree that produced the
shipped model. We copied only the files the build imports and kept the folder
layout, so the relative paths still work.

| file | what it does |
|------|--------------|
| `sleep/build.py` | builds both matrices for a window, cached in `cache/sleep/` |
| `sleep/windows.py` | the NREM periods of at least 25 min |
| `sleep/nightraw.py` | the whole night's two channels, cached in `cache/night/` |
| `common.py` | `proba`: model 1's probability for a matrix |
| `chains25.py`, `norms.py`, `norm/norms6.py`, `feats.py`, `score.py` | causal chain, z-scores, scoring |
| `_ext/cycles/hypno.py` | the subject table, and hypnogram to NREM periods |
| `_ext/cycles/cache_night.py`, `_ext/mf7/mff_direct.py` | read the `.mff` without MNE |
| `_ext/bin20/`, `_ext/simulator-without-cleaning/rt/` | the offline chain and the causal filters |
| `_ext/simulator-without-cleaning/spm_features_cell15.py` | the 210 features |

## Changes from the original

* `spm_features_cell15.py` is cell 15 of `spm_extraction_annotated.ipynb`, the
  code unchanged. `pipeline.py` used to run it from the notebook, and the rest
  of that notebook is Colab setup.
* `pipeline.py` loads model 1 from `../../models/xgb_frozen.pkl`. It is the
  same file, byte for byte, as the one it loaded before.
* `nightraw.py` falls back to `hypno.SUBJECTS` for a subject it does not know,
  so a new subject is registered in one place.
* Comments were shortened. The code is the same.

## Checked

* Rebuilding DH3 c2 from the cached night gives the same `causal` and
  `offline` matrices as the research tree, value for value.
* With the research tree's matrices for DH3, AO0 and MF7, `../train.ipynb`
  refits `../models/correction.json` byte for byte.
* Not checked: reading a `.mff` directly, since the raw files were not
  available. That code is as it was in the research tree, where
  `mff_direct.py --verify-dh3` matched MNE.

## Data

None is included. Recordings go in `_ext/mff's/`, hypnograms in
`_ext/scoring-sleep_stages/`. Caches are written to `cache/` and
`_ext/cycles/cache/`. All four are in `../../.gitignore`.

## What will not work here

Some functions the build never calls are still in the files. They need parts
of the research tree that are not included, and fail if called:

| function | needs |
|----------|-------|
| `subjects.configure`, `subjects.reference_meta` | the lab-crop caches, `ao0/`, `mf7/` |
| `common.reference`, `common.cell`, `common.teacher_x`, `common.gate` | the same, plus `bin20/cache/` |
| `feats.build_x` | `filter_variants_50ms/cache/`, `bin20/cache/` |
| `pipeline.load_baseline_times`, `pipeline.score` without `ref`, `pipeline.py --self-test` | `compare_notebook_vs_sim.py` |

The build itself (`build.py`, `windows.py`, `nightraw.py`, `common.proba`)
needs nothing outside `dev/research/`.
