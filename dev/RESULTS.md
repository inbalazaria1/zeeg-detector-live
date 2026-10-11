# Pre-training the second XGBoost on the whole night's NREM

> **Note for this repository.** This is the write-up from the research package
> this repository was collapsed out of, kept because it is the evidence: what
> was tried, what was rejected, and why the constants are what they are.
>
> **It refers to files that are not here.** Scripts (`sleep/train.py`,
> `sleep/build.py`, `common.py`, `selftest.py`, …), caches (`cache/sleep/`,
> `cache/night/`) and artefacts (`det.json`, `nrem_A.json`, `latency.log`) all
> live in the research package (the rejected variant in an earlier version of
> it), which is not distributed. Nothing here reads them.
>
> What carries over:
>
> | in this file | here |
> |---|---|
> | the shipped chain and both models | `../detect.ipynb` |
> | the recipe for model 2 (§11) | `train.ipynb` |
> | the headline numbers (§0, §14) | `../README.md` |
> | `pipeline.detect`, stage 7 | `ThresholdDetector` in `../detect.ipynb` |
> | `realtime_fir101` / `rtchain` / `rtdetect` | inlined into `../detect.ipynb` |
>
> §10b (`expandg10`) was run only in the earlier version.


**Adopt `nrem_ns` + `hold70`, fitted deterministically.** Fitting the correction
regressor on every NREM period of the two donor recordings (287 to 458 min
instead of 94 to 111) and dropping each training window's 20 min normalisation
seed raises the mean exact-epoch F1 from the frozen detector's **0.339 to
0.383** and its mean precision from **0.316 to 0.388**. The threshold stays at
0.70; the signal chain, the features, the frozen classifier and stage 7 do not
move. Turning off `subsample`/`colsample` makes the correction bit-reproducible
with no seed at all, and is worth +0.005 over the 5-seed bag it replaces (§10a).
§9 answers "precision 0.422, F1 0.339 — how is that possible?".

| | DH3 p / F1 | AO0 p / F1 | MF7 p / F1 | mean p | mean F1 |
|---|---|---|---|---|---|
| frozen booster (baseline) | 0.306 / 0.334 | 0.234 / 0.249 | 0.408 / 0.434 | 0.316 | **0.339** |
| correction on the lab crops (what existed) | 0.402 / 0.376 | 0.287 / 0.194 | 0.457 / 0.467 | 0.382 | **0.346** |
| **correction on all NREM, seed dropped** (5-seed bag) | 0.380 / 0.431 | 0.257 / 0.253 | 0.524 / 0.450 | 0.387 | **0.378** |
| **the same, deterministic single fit** (§11, ship this) | 0.368 / 0.426 | 0.260 / 0.268 | 0.536 / 0.455 | **0.388** | **0.383** |
| *the same + the patient's own later NREM (tier 3)* | *0.414 / 0.441* | *0.261 / 0.273* | *0.499 / 0.440* | *0.391* | *0.385* |

The gain is **+0.039 mean F1 and +0.071 mean precision** over the frozen
detector; **+0.032 / +0.036** over the lab-crop correction rule for rule, and
**+0.025 / +0.014** against that correction's best rule in each bag (`rate`,
0.353 and 0.365). PLAN.md's noise floor is 0.03 mean F1, and a config that wins
on one recording and collapses on another does not count. Against the frozen
detector this clears the floor and wins on all three patients, in precision and
in F1, in both bags. Against the incumbent correction it clears the floor
rule-for-rule but not against its luckiest rule, so the honest claim is *"beats
the frozen detector by 0.039, and beats the lab-crop correction"*, not *"by 0.03
over everything"*.

---

## 1. What was changed, and what was not

The detector is two models.

* **Model 1, frozen.** `no_change_run/xgb_s50_f210_b_3000max.pkl`, the
  notebook's XGBoost classifier on 210 features of the 25 ms causal chain
  `b25_o3 × expand20` (causal Butterworth order 3 0.1 Hz high-pass, 51-tap
  symmetric 70 Hz firwin read 25 samples late, trailing 2 s box detrend,
  expanding z-score seeded on the first 20 min). Not touched here.
* **Model 2, the one this folder fits.** An XGBoost **regressor** predicting, in
  logit space, `logit(teacher) − logit(frozen)`, where the teacher is model 1
  reading the **offline** chain (33001-tap zero-phase band-pass, centred 2 s
  SMA, whole-window z-score, 16.5 s of lookahead). Applied as
  `sigmoid(logit(frozen) + α·correction)`; α = 0 reproduces the frozen detector
  exactly. Bagged over 5 seeds, because one draw has sd 0.017 on a 0.34
  statistic (`model/bag_stack.py`).

Only **what model 2 is fitted on** changed:

| set | training data per fold | tier |
|---|---|---|
| `lab` | the two donors' lab crops — 94 to 111 min, 22–27 k rows | 1 |
| `nrem` | every NREM period ≥ 25 min of the two donors — 287 to 458 min, 69–110 k rows | 1 |
| `nrem_ns` | the same windows with each one's own 20 min seed block dropped — 35–62 k rows | 1 |
| `nrem_own` | `nrem` + the test patient's own NREM periods that do not touch its lab crop | 3 |

Everything else is held: the frozen classifier, the signal chain, the features,
`pipeline.detect` (threshold, 2 s refractory, 20 min bins), and the metric. The
test data is the published lab-crop matrix from an earlier experiment's cache,
and the reference is `common.reference()`, so every row here is comparable to
the baseline table in `../README.md`. `common.gate()` PASSED at the top of both
runs.

### Where the extra data comes from

A note written alongside the sleep staging (not distributed) turns the lab's expert
hypnograms into NREM periods. The three lab crops are **one NREM period each**
(DH3 cycle 1, AO0 cycles 1+2, part of MF7 cycle 1) and use 154.5 of the 744 min
of NREM sleep in the three recordings, i.e. **21 %**. Periods under 25 min are
unusable, because `expand20` spends the first 20 min seeding. That leaves
584 min of training material:

```
        lab crop                 long NREM periods                   donor pool
DH3     23 - 74     (51.0)       c1 51.2  c2 42.2  c4 33.1              126.5 min
AO0      4 - 47.5   (43.5)       c3 46.7  c4 40.1  c5 42.2  c6 31.1     160.1 min
MF7      2 - 62     (60.0)       c1 79.3  c2 61.2  c3 26.1  c4 52.7
                                 c5 49.2  c6 29.1                       297.7 min

per fold:  test DH3  457.8 min (4.4x)   test AO0  424.2 min (3.8x)   test MF7  286.6 min (3.0x)
```

MF7's whole night had to be downloaded first: the earlier cache stopped at minute
173 and cycles 4-6 are past it (`sleep/nightraw.py`, 3 s on a warm mount).

### That the builder is the same builder

Everything rests on the claim that a matrix built over an arbitrary NREM period
is what the published pipeline would have produced. `sleep/selftest.py` tests it
by pointing the new builder at the window the old one ran on:

```
1. the night cache, sliced, == the per-window .npz caches       3/3 identical
2/3. the lab crop rebuilt through sleep/build.py
     DH3 causal / offline   max|diff| = 0     (bit-identical)
     AO0 causal / offline   max|diff| = 0     (bit-identical)
     MF7 causal / offline   5.1e-09 / 1.6e-07
4. detect(teacher) over the lab crop == common.reference()      409 / 195 / 839
5. the rebuilt matrices produce the SAME detections             6/6 identical
```

MF7's two are not bit-identical because its published raw cache is one sample
shorter than a `[0, 63] min` slice of the night, which moves the 33001-tap FIR's
FFT blocking; §5 shows it changes no detection. DH3 and AO0 match exactly.

---

## 2. The threshold rule is the whole result

More data made the **ordering** of epochs much better and, at first, made the
deployed detector **worse**. The gap between the two is one number: where the
threshold goes. The correction is added in logit space, so the corrected score
does not live on the scale 0.70 was chosen for.

Four rules were tried, all complete recipes a device could execute.

| rule | what it is allowed to know |
|---|---|
| `hold70` | α and depth maximise pooled F1 on the donors' NREM periods **at th = 0.70**; deploy at 0.70. α = 0 is in the grid, so this can always fall back on the frozen booster. |
| `donorF1` | α, depth **and** th all maximise pooled F1 on the donors' NREM periods. |
| `rate` | α, depth from `donorF1`; th re-anchored so the corrected detector fires as often inside **this patient's own 20 min normalisation block** as the frozen one does there at 0.70. |
| `qcal` | the same, matching the *fraction of epochs* above threshold rather than the detection count. |

Mean F1, seed set A / seed set B:

| training set | `hold70` | `donorF1` | `rate` | `qcal` | oracle (tier 3) |
|---|---|---|---|---|---|
| frozen baseline | 0.339 | 0.339 | 0.339 | 0.339 | 0.348 |
| `lab` | 0.346 / 0.343 | 0.347 / 0.337 | 0.353 / 0.365 | 0.343 / 0.365 | 0.388 / 0.388 |
| `nrem` | **0.368 / 0.369** | 0.333 / 0.366 | 0.322 / 0.313 | 0.333 / 0.301 | 0.401 / 0.388 |
| `nrem_ns` | **0.378 / 0.379** | 0.364 / 0.365 | 0.366 / 0.341 | 0.372 / 0.368 | 0.420 / 0.418 |
| *`nrem_own`* | *0.385 / 0.388* | *0.334 / 0.337* | *0.316 / 0.294* | *0.345 / 0.309* | *0.414 / 0.398* |

**Every rule that moves the threshold loses to the rule that does not** in six
of the eight (set, bag) pairs, and in all six with an NREM training set. The two
exceptions are both the small `lab` set, where `rate` wins by 0.007 and 0.022,
which is the row `model/RESULTS.md` already reported as the best deployable one.
The more data the correction saw, the worse moving the threshold gets. `donorF1`
on `nrem` bag A picks 0.60 for DH3 and turns 175 false positives into 492: the
donors' NREM periods want a lower threshold than the lab crop does, and that
preference does not transfer. The own-block anchors fail from the other side.
AO0's normalisation block has **three** detections, so `rate` is fitting a
threshold to a count of three, and MF7's `rate` pick of 0.47-0.51 costs it 0.13
to 0.16 F1.

The earlier cross-cycle experiment called this a gain error: another stretch of the night
carries the same information at the wrong gain. The answer is to never leave the
gain. Hold 0.70 and let the inner loop decide only how much correction to add.
`hold70`'s α picks are 1.00 / 0.15-0.25 / 0.40-0.60: DH3 takes the correction
whole, AO0 takes almost none of it.

---

## 3. The winning row, in full

`nrem_ns` × `hold70`, exact epoch, threshold 0.70, scored from second 1200. A
detection is a TP only if it lands on the same 250 ms epoch the offline notebook
chain fired on.

**seed set A**

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| DH3 (261 ref) | 130 | 212 | 131 | 0.380 | 0.498 | **0.431** |
| AO0 (185 ref) | 46 | 133 | 139 | 0.257 | 0.249 | **0.253** |
| MF7 (495 ref) | 195 | 177 | 300 | 0.524 | 0.394 | **0.450** |
| mean | | | | **0.387** | **0.380** | **0.378** |

**seed set B** (no seed in common with A)

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| DH3 | 131 | 229 | 130 | 0.364 | 0.502 | **0.422** |
| AO0 | 50 | 146 | 135 | 0.255 | 0.270 | **0.262** |
| MF7 | 197 | 181 | 298 | 0.521 | 0.398 | **0.451** |
| mean | | | | **0.380** | **0.390** | **0.379** |

**against the frozen detector** (identical scoring, identical threshold)

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| DH3 | 96 | 218 | 165 | 0.306 | 0.368 | 0.334 |
| AO0 | 49 | 160 | 136 | 0.234 | 0.265 | 0.249 |
| MF7 | 230 | 334 | 265 | 0.408 | 0.465 | 0.434 |
| mean | | | | 0.316 | 0.366 | 0.339 |

Per patient (bag A vs baseline):

* **DH3** +0.097 F1. 34 more true positives on 6 fewer false positives. The
  correction is doing real work here, and DH3 takes it at α = 1.0.
* **MF7** +0.016 F1 but **+0.116 precision**: 157 false positives deleted for 35
  true positives given up. If precision is what the closed loop cares about,
  this is the largest single change in the table.
* **AO0** +0.004 F1, inside the noise. AO0 is the recording where nothing has
  ever moved much; its normalisation block has 3 detections and its lab crop
  straddles a stage transition (`project-ao0-window`).

The gain is **not** the bagging and **not** the seed: the two disjoint bags agree
to 0.001 mean F1, against the 0.017 sd of a single unbagged seed.

---

## 4. Dropping the seed block is worth +0.010, and why

`nrem` → `nrem_ns` is the same windows with rows `t < 1200 s` removed from the
TRAINING data (the test window is untouched). It costs 44 % of the rows and buys
+0.010 mean F1 in both bags.

`model/RESULTS.md` §regime found the reason before this experiment existed. In
`expand20` the six `chan_*` columns are numerically **constant inside the seed
block** (sd 0 to 3e-14), because `build_epochs` pins `ends = max(starts + EPOCH,
k)` for every seed epoch, and they vary by O(1)–O(18) outside it. So 44 % of the
training rows were teaching the regressor a normalisation regime the detector
never runs in. With one lab crop per donor you cannot afford to drop them; with
10 windows you can.

---

## 5. What the patient's own night adds: almost nothing

`nrem_own` adds the test patient's own NREM periods that do not touch the crop
being scored: tier 3 here, since it reads later parts of tonight, and **tier 2**
in a clinic that records the same patient twice. It is worth **+0.007 / +0.009
mean F1** over `nrem_ns`, and it is not uniformly better — MF7 loses 0.010 in
bag A.

That is the useful negative: **the correction does not need to know the
patient.** Two other recordings' whole nights are as good as adding this
patient's own, so a device shipped with a donor-fitted correction does not have
to wait a night to be worth using.

(Both `nrem_own` rows in the first version of this run were contaminated by a
corrector cache keyed on the set of *subjects* rather than the set of *windows*.
With all three recordings in every fold the key collided, and one fold got a
model fitted on the window it was being tested on. Fixed in `train.py`; the
numbers above are the fixed ones.)

---

## 6. What did not work, so it is not tried again

* **Letting the inner loop choose the threshold** (`donorF1`). Loses 0.014 to
  0.035 to holding 0.70, and the loss grows with the amount of training data.
* **Anchoring the threshold on the patient's own normalisation block**, by
  detection count (`rate`) or by epoch fraction (`qcal`). Both lose. AO0's block
  has three detections; MF7's anchored threshold is 0.19 below its oracle. This
  is the third folder to find that the normalisation block cannot set a
  threshold (`PLAN.md`, `project-realtime-fir101-simulator`).
* **NREM periods shorter than 25 min.** `expand20` seeds on 20 min and
  `pipeline.detect` drops a final bin under 10 min, so DH3 c3/c5/c6, AO0
  c1/c2/c7 and MF7 c7/c8 contribute no scorable epoch at all. MF7 c3 and c6 are
  long enough to train on but produce **0** scored reference spikes, which is
  why the inner validation pool is filtered on `n_ref >= 20` and not on
  duration.
* **α as a free parameter with the threshold.** The two are nearly redundant, in
  that a bigger α and a lower threshold buy the same detections, and giving the
  inner loop both is what lets it walk off the operating point.

---

## 9. Why the published `expandg10` row has precision 0.422 and F1 0.339

An earlier trial's threshold sweep reports, for the same
chain with `expandg10` and the lab-crop correction at th 0.70:

```
 0.70   DH3 0.436/0.272/0.335   AO0 0.310/0.168/0.218   MF7 0.521/0.418/0.464
        mean p 0.422   mean r 0.286   mean F1 0.339
```

Reproduced here to within the α-selection grid (`g_bagA.log`, row `0.70j`):
mean p 0.407, r 0.293, F1 0.338. Three things are going on and only the third
is a problem.

**(a) The precision did go up. The recall paid for it.** Against the frozen
booster's 0.316 / 0.366 / 0.339, that row is **+0.106 precision and −0.080
recall**. F1 is the harmonic mean of the two, so it does not move. The detector
was moved up its own precision-recall curve.

**(b) Precision alone is not comparable across configurations,** because any of
these detectors will hand you more precision if you raise the threshold. The
`meanP` line of that sweep is flat at 0.39-0.42 all the way from th 0.66 to th
0.82. So "precision 0.422" says where the threshold sits, not how good the model
is. The comparable question is *what precision is reachable at a given recall*,
and by that measure more data helps everywhere (`compare.py`, interpolated
precision):

| best precision reachable at recall ≥ | 0.25 | 0.30 | DH3 @ 0.45 | MF7 @ 0.45 |
|---|---|---|---|---|
| frozen booster | 0.362 | 0.333 | unreachable | 0.455 |
| correction on the lab crops (`expandg10`) | 0.415 | 0.384 | unreachable | 0.514 |
| **correction on all NREM, seed dropped** | **0.424** | **0.402** | **0.424** | **0.556** |

The frozen booster **cannot reach recall 0.45 on DH3 at any threshold**. The
NREM-trained correction reaches it at precision 0.424. That is what more input
bought, and it is invisible at a single operating point.

**(c) The real problem: the correction moves the SCALE, and 0.70 was chosen for
the frozen scale.** Adding `α·correction` in logit space shifts the whole score
distribution down, so at a fixed 0.70 the detector fires less. In that published
row α was chosen jointly with a threshold of 0.60-0.65 and then *deployed at
0.70*: tuned for one operating point, run at another. Fix that (choose α on the
donors **at th = 0.70**, the `hold70` rule) and the same correction family goes
from F1 0.339 to 0.383.

**So: more input did help. It just could not show at a threshold the correction
had been allowed to walk away from.**

---

## 10. What was tried to push it further

Both are deterministic and both are selected on donors only.

| change | mean p | mean r | mean F1 | verdict |
|---|---|---|---|---|
| `nrem_ns`, 5-seed bag (§3) | 0.387 | 0.380 | 0.378 | the previous best |
| **`nrem_ns`, deterministic single fit** | **0.388** | **0.392** | **0.383** | **adopt** |
| `expandg10` training rows instead of dropping the seed | 0.359 | 0.373 | 0.365 | reject |

### 10a. Determinism, and it is free

`subsample` and `colsample_bytree` were 0.8, the only stochastic part of the
booster and the only reason the 5-seed bag existed. Setting both to 1.0 makes
the fit **bit-identical across seeds and across reruns**, checked at the booster
level (`selftest.py` §7: seed 1 vs seed 2 max|diff| = 0, against 0.212 at
subsample 0.8) and end to end (a full rerun produces byte-identical threshold
curves and probability vectors).

It is not a compromise. It scores **0.383 against the bag's 0.378**, needs one
fit instead of five (95 s for all three folds instead of ~1000 s), and leaves no
seed list to carry around. This is the row to ship.

### 10b. `expandg10` is not the way to repair the seed block — reject

`expandg10` and `expand20` are **bit-identical on every scored epoch**
(`selftest.py` §6: max|diff| = 0 on epochs ≥ 4800, and 1e5–1e7 inside the seed).
So `expandg10` is the same detector with the normalisation block's 4800 rows
*repaired* instead of frozen. That is the same disease `nrem_ns` treats by
*deleting* those rows, and deleting wins: 0.383 against 0.365. Even normalised
properly, the first 20 min of a window are a transient the detector never
operates in, and they are 44 % of the rows.

Corollary worth keeping: **`nrem_ns` is norm-independent.** With the seed
dropped the two training sets are the same rows, so there is nothing left to
choose between `expand20` and `expandg10`.

---

## 11. The model to ship

```
stage 1   causal Butterworth order 3 0.1 Hz high-pass          0 ms
          51-tap symmetric 70 Hz firwin, read 25 samples late  25 ms
stage 3   trailing 2 s box detrend                             0 ms
stage 6   expanding z-score seeded on the first 20 min (`expand20`)
model 1   the frozen notebook booster on the 210 features      UNCHANGED
model 2   XGBoost reg:squarederror on those 210 features + the frozen logit,
          target logit(teacher) - logit(frozen),
          eta 0.05, 300 rounds, min_child_weight 5, reg_lambda 2,
          subsample 1.0, colsample_bytree 1.0        <- deterministic, no seed
          trained on EVERY donor NREM period >= 25 min, each window's first
          20 min dropped
          depth and alpha chosen by leave-one-donor-out on the donors' own
          NREM periods, scored at th 0.70:
              DH3  depth 6  alpha 1.00     (10 windows, 458 min, 61 841 rows)
              AO0  depth 6  alpha 0.15     ( 9 windows, 424 min, 58 574 rows)
              MF7  depth 6  alpha 0.80     ( 7 windows, 287 min, 35 171 rows)
score     sigmoid(logit(frozen) + alpha * correction)
stage 7   pipeline.detect, threshold 0.70, 2 s refractory, 20 min bins
```

Exact epoch, scored from second 1200:

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| DH3 (261 ref) | 132 | 227 | 129 | 0.368 | 0.506 | **0.426** |
| AO0 (185 ref) | 51 | 145 | 134 | 0.260 | 0.276 | **0.268** |
| MF7 (495 ref) | 196 | 170 | 299 | 0.536 | 0.396 | **0.455** |
| **mean** | | | | **0.388** | **0.392** | **0.383** |
| frozen booster, same scoring | | | | 0.316 | 0.366 | 0.339 |
| the published `expandg10` row | | | | 0.422 | 0.286 | 0.339 |

**+0.044 mean F1 over the row in the question**, at +0.106 recall for −0.034
precision. 0.70 is the right place to stand: it is where this model's F1 peaks
*and* where its precision has already plateaued.

```
  th      0.66   0.68   0.70   0.72   0.74   0.76   0.78   0.80
  mean p  0.332  0.361  0.388  0.388  0.392  0.391  0.393  0.385
  mean r  0.403  0.402  0.392  0.334  0.280  0.216  0.157  0.107
  mean F1 0.355  0.372  0.383  0.353  0.323  0.276  0.223  0.164
```

So there is **no precision to be bought by raising the threshold**, on this
model or on the published one. Precision above ~0.39 is not a threshold problem.

### On overfitting

Nothing that touches the scored night is used. Per fold the correction sees only
the other two recordings, and depth, α and the threshold rule are all chosen by
leave-one-donor-out *inside* those two. The only hyper-parameters selected at
all are depth (2 values) and α (7 values), on 7–10 donor windows, with the
threshold nailed — fewer knobs than the `joint` rule, which has a third axis and
loses 0.02–0.06 to this one. Every rule that gave itself more freedom did worse,
which is the signature of selection overfitting rather than under-modelling.

---

## 12. Reproducing

**None of these commands run in this repository.** They are the research
package's, kept so the numbers above have a provenance. They run from the
research package's root (26 cells, `expandg10` removed):

```bash
./.venv/bin/python sleep/windows.py                     # the NREM period plan
./.venv/bin/python sleep/build.py   --plan              # which matrices exist
./.venv/bin/python sleep/build.py   --all               # 26 cells, ~45 min
./.venv/bin/python sleep/selftest.py                    # all sections ok

# the row to ship - deterministic, 95 s, no seed
RETRAIN_NTHREAD=8 ./.venv/bin/python sleep/train.py \
    --sets lab,nrem,nrem_ns --det --out sleep/det.json

# the earlier 5-seed bag
RETRAIN_NTHREAD=8 ./.venv/bin/python sleep/train.py \
    --sets lab,nrem,nrem_ns,nrem_own --seed-set A --out sleep/nrem_A.json

./.venv/bin/python sleep/compare.py                     # the PR frontier, §9
```

In the earlier version, for the rejected variant:

```bash
RETRAIN_NTHREAD=8 ./.venv/bin/python sleep/train.py \
    --norm expandg10 --sets lab,nrem --out sleep/g_bagA.json          # §10b
```

Artefacts kept in the research package: `det.json` / `det.log` / `det.npz` (the
shipped row) and `nrem_A.json` (the 5-seed bag `compare.py` reads for its two
middle rows). The `.json` holds the full 0.30–0.98 threshold curve of every cell
and `det.npz` every corrected probability vector, so a new threshold rule costs
no refit. The run transcripts, the second seed bag and the rejected variant's
artefacts were dropped as re-derivable or parent-only. **None of these files are
in this repository**; the only artefacts here are the two models in
`../models/`.

Runtime: ~45 min to build the 26 feature matrices (one-off, cached in
`cache/sleep/`), then 95 s for the deterministic run and 15-18 min per 5-seed
bag.

## 13. What is written where

*(Of the research package, where these runs happened.)* It writes only inside
its own folder; the earlier experiments and the raw recordings it reads are
read only. MF7's whole-night raw cache is written into its `cache/night/`.

---

## 14. Measured latency (`latency.py`, `latency.log`)

The earlier real-time simulator (`realtime_fir101.py`) imported read-only and reconfigured in memory for
this chain: 51-tap firwin (25 ms) behind a causal **order-3** 0.1 Hz
Butterworth, with the correction booster spliced into its timed loop. Both
boosters go through `Booster.inplace_predict` on a preallocated row, because the
sklearn `predict_proba` wrapper spends ~49 ms per call validating feature names
and would swamp the measurement. Feed is the in-memory raw cache, so no disk
I/O is charged.

Per 250 ms decision, pooled over the three lab crops (22 671 timed windows):

| | median | p95 | p99 |
|---|---|---|---|
| filter + detrend + normalise | 0.17 | 0.22 | 0.35 |
| feature extraction (210 features, 2 channels) | 10.17 | 11.00 | 12.8 |
| **model 1** (frozen booster) | 0.40 | 0.54 | 0.72 |
| **model 2** (the correction) | **0.14** | **0.19** | **0.32** |
| compute | 10.89 | 11.86 | 14.1 |
| **end-to-end** (25 ms lookahead + compute) | **35.9** | **36.9** | **39.1** |

**The second model costs 0.14 ms**, 1.3 % of the compute and 0.4 % of the
end-to-end latency. Feature extraction is 93 % of the compute and always was.
Adding the correction does not change the deployability of the detector.

Budget: 2 of 22 671 windows exceeded the 250 ms epoch, both on AO0, both feature
extraction stalls under load (max 234 ms). Throughput 34-41× real time on 8
cores.

### Cross-check against the batch harness

The simulator's own detection count over the scored span, against the batch
number every table above is computed from:

| | streamed | batch |
|---|---|---|
| DH3 | 359 | 359 |
| AO0 | 196 | 196 |
| MF7 | 366 | 366 |

**Exact on all three.** The streaming chain and the two models are the ones this
folder measured, detection for detection.

This took one fix. AO0 used to stream 274 against the batch 196, and an earlier
version of this section blamed `realtime_fir101.Expanding`, a running
accumulator, for drifting away from `norms.expand20`'s cumulative-sum
recomputation. **That diagnosis was wrong.** Measured on AO0, the two
normalisers agree to 1e-14 relative on the mean and the sd, and the detrended
samples are bit-equal.

The whole gap was one line of stage 7. `pipeline.detect` skips a bin shorter
than half a bin (`if len(p) < stride // 2: continue`), and
`realtime_fir101.Detector` has no such rule. AO0's crop is 43.5 min = 2.17 bins,
so its final 3.5 min bin is dropped by the batch scorer and kept by the
streaming one: 82 extra detections, all after minute 40. DH3 (2.55 bins) and MF7
(exactly 3.00) have no short final bin, which is why they always agreed.

`rtdetect.Detector` applies the floor and reproduces `pipeline.detect` exactly on
all three (`rtdetect.py`). It also takes `drop_short_final=False`, which is what
a live device should use: the rule exists so a partial bin cannot report a
spikes-per-minute figure computed over too little signal, and a device that
keeps running has no final bin until someone stops it.
