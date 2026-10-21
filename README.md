# zEEG detector: near-real-time IED detection in sleep EEG

Interictal epileptiform discharge detection from an EGI `.mff`, fast enough to
drive closed-loop stimulation. It takes ~36 ms to get from the end of a 250 ms
window to a decision: 25 ms of lookahead inside the band-pass, ~11 ms of
compute, nothing else. The budget is **50 ms** — the interval within which a
stimulus still falls inside the window the detected discharge defines — so there
are ~14 ms to spare. The 250 ms epoch is a separate constraint: it bounds
throughput, not latency.

The detector is one notebook, [`detect.ipynb`](detect.ipynb), plus the two
models in `models/` and one file of constants, [`constants.py`](constants.py).
There is no package to install and no module tree to read. Only python3.9 and requirements. 

```bash
python3.9 -m venv .venv
./.venv/bin/pip install -r requirements.txt jupyterlab
./.venv/bin/jupyter lab detect.ipynb
```

Every constant the two notebooks share is defined once, in `constants.py`, which
`dev/train.ipynb` imports too. The two notebooks can't
disagree about the threshold, the epoch length or the calibration block, because
there is only one definition of each.

Edit the parameters cell (the `.mff` path and the minutes to analyse) and Run
All. Output is `(time_sec, time_min, probability)` per detection, written to a
CSV (or printed, with `OUTPUT_CSV = None`).

### On Colab

Uploading only `detect.ipynb` will not work: the two model files are 2.4 MB and
are not embedded in the notebook. Get the whole repository first, in one cell:

```python
!git clone <this repository's url>
!pip install -r zeeg-detector/requirements.txt
```

(`zeeg-detector/` is the folder the clone creates; use the repository's name if
it differs.)

Then **Runtime → Restart session**, which installing the pinned `numpy`/`scipy`
over Colab's own versions requires, and Run All. You do not need `%cd`: the
notebook locates its own directory instead of trusting the working directory.

Three things to expect there, none of them fatal:

* **Colab's Python is newer than 3.9.6**, which is what these numbers were
  measured on. The pins in `requirements.txt` should still resolve, but if `pip`
  cannot satisfy one, the notebook's version banner will say which. The
  synthetic fingerprint in the verification section is what tells you whether it
  mattered. Treat a fingerprint mismatch as "these are not the published
  numbers", not as a bug to work around.
* **It will be slower.** The 11 ms compute figure is 8 cores; a free Colab VM has
  2. Feature extraction is 93 % of the compute, so expect roughly 2–4× longer
  wall clock. The latency numbers the notebook prints then measure Colab, not a
  device.
* **Your recording has to get there.** Mount Drive
  (`from google.colab import drive; drive.mount('/content/drive')`) and point
  `MFF_DIR` at the `.mff` directory. Do not commit it, and think before uploading
  patient data to a Google VM. That is a decision about your data, not a
  technical step.

---

## What it does

| | |
|---|---|
| input | an EGI `.mff` **directory** at 1000 Hz, channels **E227** and **E254** |
| output | `(time_sec, time_min, probability)` for every detected discharge |
| decision rate | one per 250 ms epoch |
| latency | **35.9 ms** median, 36.9 ms p95, per decision, against a **50 ms** budget |
| warm-up | the first **20 min** seed the z-score and produce no detections |
| throughput | ~11 ms of compute per 250 ms decision, a ~4 % duty cycle. Replaying a file runs 20–40× real time, depending on I/O and load |

The detector never holds the recording. It holds a `SampleFeed` whose only method
is `take(n_samples)`, so there is no call that returns a sample the wall clock has not
reached yet. That is what makes "no lookahead beyond 25 ms" something you can
check rather than a promise. Swap that `SampleFeed` for one backed by a live amplifier
and nothing else in the chain changes; the notebook keeps the seam in its own
cell, with a sketch and a list of what live operation still needs.

## The chain

```
stage 1   causal Butterworth order 3 0.1 Hz high-pass            0 ms
          51-tap symmetric 70 Hz firwin, read 25 samples late   25 ms
stage 3   trailing 2 s box detrend                               0 ms
stage 6   expanding z-score, seeded on the first 20 min          0 ms
model 1   the frozen notebook booster, 210 features   (never refit)
model 2   XGBoost reg:squarederror on those features + the frozen logit,
          target logit(teacher) - logit(frozen)
score     sigmoid(logit(frozen) + alpha * correction)
stage 7   threshold 0.70, 2 s refractory, 20 min bins
```

Model 2 is the contribution. It learns, from past recordings, how far the causal
chain's score sits from what the same classifier would have said with 16.5 s of
future available, then adds that back. The offline 16.5 s chain only generates
training labels. It is never called at inference. Its code is in
`dev/research/`, which you only need to refit model 2.

## How well it works

Leave-one-recording-out over the three recordings it was built from, threshold
0.70:

| | precision | recall | F1 |
|---|---|---|---|
| frozen classifier alone | 0.316 | 0.366 | 0.339 |
| **+ the NREM-trained correction** | **0.388** | **0.392** | **0.383** |

Per recording: DH3 0.426, AO0 0.268, MF7 0.455. **Expect this, not better, on a
recording the model has not seen.** The correction costs 0.14 ms per decision,
0.3 % of the 50 ms budget.

`dev/RESULTS.md` is the full write-up: what was tried, what was rejected, and why
the threshold must stay at 0.70.

Only model 2 can be refitted, and only when new donor recordings turn up. The
recipe is `dev/train.ipynb`, explained in `dev/README.md` ("Adding a new
subject" is the step-by-step). It needs the raw recordings, or matrices built
from them, for the original three donors as well as the new one. Those are not
distributed, so it will not run from a clone of this repository.

---

## Read this before you use the numbers

* **Agreement, not ground truth.** Every number above is agreement with an
  offline reference chain (33001-tap zero-phase band-pass, 16.5 s of lookahead)
  over the same recording. **No clinician annotated any of this.** Scoring is
  exact epoch: a detection counts only if it lands on the same 250 ms epoch the
  offline chain fired on. 
* **Three recordings**, all EGI 1000 Hz, all N3-heavy, all from the lab. 
* **The shipped model has no held-out patient.** Its `depth` and `alpha` were
  selected on all three recordings, because a new patient is the fourth. The
  honest performance estimate is the leave-one-recording-out table above, not
  anything measured on the model in `models/`.
* **Channels are hard-wired** to E227/E254 at 1000 Hz. Another montage or another
  sampling rate means a retrain, not a config change.
* **20 min of warm-up per night is structural.** Stage 6 cannot normalise a
  patient it has not seen, and 20 min is the value everything was fitted at.

### Do not tune the threshold

Four rules that move it off 0.70 were tried and all four lose (`dev/RESULTS.md`
§2 and §6), including fitting it on the patient's own calibration block.
Precision above ~0.39 is not reachable by thresholding: mean precision is flat
from 0.70 to 0.80 while recall collapses. The shipped correction was selected at
0.70, so moving the threshold invalidates that selection as well as the numbers.

---

## Layout

```
detect.ipynb          the whole detector                        <- start here
constants.py          every constant, defined once, shared by both notebooks
models/
  xgb_frozen.pkl        model 1, the frozen classifier (420 KB)
  correction.json       model 2, the correction booster (2.0 MB)
  correction_meta.json  what it was fitted on, and its alpha
requirements.txt
.gitignore
dev/                  NOT needed to run - for future changes
  README.md             what is in here and when you would need it
  train.ipynb           refit the correction on new donor recordings
  research/             recording to training matrices, used by train.ipynb
  RESULTS.md            the write-up: what was tried, what was rejected
  figures/              figure scripts and the data behind them
```

No patient data is in this repository, and none should ever be committed.
`.gitignore` excludes `.mff`, `.fif`, `.edf` and the detector's own CSV output.

## Verifying

The last section of the notebook checks itself. The results below are saved in
the file, so you can see them without running it.

| check | needs | result when this was built |
|---|---|---|
| synthetic end-to-end fingerprint — both filters, all 210 features, both boosters, stage 7 | `models/` + `constants.py` | reproduced |
| acceptance test on a real recording, frozen classifier alone | the `.mff` | **291 detections, times identical** to the package this was collapsed from |
| the same, with the correction on | the `.mff` | **267 detections, times identical** |
| the fast feature extractor vs the original notebook definitions | nothing | max abs difference **0.0** |
| `constants.py` vs the shipped model's own metadata — threshold, refractory, bins, calibration, lookahead, feature count | `models/` | agree (asserted at load) |

The acceptance test skips itself with a clear message when the recording is
absent, which it will be for anyone outside the lab: the recordings are patient
data and are not distributed. The synthetic check covers the whole chain without
one. If you do have the recording, set `ACCEPTANCE_MFF` in the acceptance-test cell
to its `.mff` directory. It is not in the Parameters cell.

If the version banner in the notebook's imports cell reports anything off the
pinned set, expect the fingerprint to differ. That is the pin doing its job, not
a bug.

After refitting the correction, the 267-detection result and the synthetic
fingerprint will most likely change, because they depend on `correction.json`.
The 291-detection frozen-only result must not change. Re-record the other two
here once you have checked them.

The isolation claim was tested rather than assumed. The folder was copied
somewhere with no siblings, a fresh venv was built from `requirements.txt` alone,
and the notebook ran end to end there: synthetic fingerprint reproduced exactly,
acceptance test skipped with its message, feature check 0.0.

## Constants

`constants.py` is the only place a number is defined. Two groups are left out of
it on purpose:

* **Feature parameters:** `emb=2`, `q=0.75`, `kmax=10`, the six band edges, the
  zero-crossing epsilon. They stay inline in `extract_epoch_features`, which is
  lifted byte-for-byte from the code the frozen classifier was trained against.
  Renaming inside it would break the one property that makes the 291-detection
  match meaningful, which is that the code was never retyped.
* **Arbitrary values in self-checks:** the synthetic-noise seed, the dummy
  `chan_ptp` in the model warm-up. There is nothing to name; they only have to
  stay the same between runs.

Nothing in `constants.py` is a knob. Each value is a property of the recording
format, a unit conversion, or the value the frozen model was trained at. The
notebook asserts the ones the shipped correction depends on against
`models/correction_meta.json` at load time, so a changed constant fails loudly
instead of quietly producing different numbers.

## Next steps

None of this is implemented. Three tracks: the streaming code, the device, and
the validation. Ordered inside each track by what depends on what.

### A. Make the streaming path live-safe

The items listed in `detect.ipynb`'s `SampleFeed` cell. All of them sit behind
the feed, so none touches the signal chain or the models.

1. **A timestamp on every block.** `take` returns the samples plus the index of
   the first one, so a gap becomes visible at all. Everything below needs it.
   Free in `MffFileFeed`, which already tracks `next_sample`; live it comes from
   the amplifier's packet header.
2. **Gap detection, and a policy for what to do.** Compare the expected next
   index against what arrived. On a gap: clear the filter state, advance the
   epoch counter by the missing samples so later detection times stay true, and
   decide the z-score rule (keep the running stats below some gap length,
   re-seed above it). Write the cutoff down with the reason. Mark the affected
   span unscored in the CSV instead of skipping it silently. Test it by cutting
   a hole in the synthetic signal in the self-check and asserting the detections
   after the hole match the no-hole run.
3. **An unbounded loop.** `run` becomes `while True` over the feed. `n_epochs`
   is then only used by `drop_short_final`, which a live device sets to `False`
   anyway.
4. **A feed that blocks at end of file.** `MffFileFeed` waits and retries
   instead of raising `EOFError`, so the whole path can be dry-run against a
   file that is still being written.
5. **Error recovery.** Wrap the per-epoch body: log the exception with its epoch
   index, reset what needs resetting, carry on. Exit only when the feed itself
   is gone.

Item 2 is the one to do first. A gap is the only failure here that produces
quietly wrong output instead of an exception.

### B. Connect to the zEEG

1. **Get the device's streaming API on paper.** Four things decide how much of
   track A is needed: the sampling rate, the packet size, whether packets carry
   a sequence number or a timestamp, and the units. The chain works in volts
   (`MffFileFeed` multiplies by `MICROVOLTS_TO_VOLTS * gain`), so a device
   handing over ADC counts converts in its own feed, not in the chain.
2. **Record a night on the zEEG and replay it through the file path.** A
   `ZeegFileFeed` beside `MffFileFeed`, no live code and no threads. This
   answers the question that matters, whether the model transfers to this
   hardware, using a pipeline that is already verified.
3. **Check the montage.** The largest risk in this list, and not an engineering
   one. The chain is hard-wired to E227/E254 on an EGI net at 1000 Hz. If the
   zEEG's electrodes do not see close to the same signal, the models do not
   apply and it is a retrain, not a config change. Best evidence is one subject
   recorded on both systems at once. A sampling rate other than 1000 Hz is the
   same conversation.
4. **Implement `LiveFeed`.** Fill in the sketch at the bottom of the
   `SampleFeed` cell, with the amplifier's callback calling `push`. A1 and A2
   first: the sequence numbers this needs are what they add.
5. **Re-measure the latency on the machine it will run on.** The budget is
   50 ms, and the numbers in the notebook are from 8 laptop cores. A dedicated
   box next to the bed will most likely be faster, not slower, but the timing
   table has to be re-run there before anyone quotes it.
6. **Start and stop signals.** Processing needs an explicit start, so the 20 min
   normalisation block is seeded on the right stretch of the night, and an
   explicit stop, so the detector knows the session is over and does not sit
   waiting on a feed nobody is filling. A drop-out and reconnect is the awkward
   case: it is gap handling again, with a gap longer than the re-seed cutoff.

### C. Validation

Track A makes the code trustworthy; it does not make the detector trustworthy.
See "Read this before you use the numbers" above.

1. **Clinician-marked spikes on at least one recording.** Until then every
   number here is agreement with an offline chain, not detection of spikes.
2. **A fourth patient, held out for real.** Fit nothing on it, report that
   number, treat it as the honest one.

Across the tracks: A1, A2, then B2 and B3, then the rest. B2 and B3 come early
because if the montage does not transfer, the live plumbing was the wrong thing
to build first.
