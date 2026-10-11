# Data export brief — figures for the near-real-time zEEG manuscript

You are exporting **already-computed results** from an existing analysis pipeline so that
figures can be drawn from them. You are **not** re-running science, not retraining, and not
changing any number.

---

## 0. Context

The project adapts a validated offline zEEG epileptiform-discharge (ED) detector for
near-real-time operation. A frozen XGBoost classifier scores every 250 ms epoch; a strictly
causal signal chain feeds it; a second XGBoost model ("the correction") adds back, in logit
space, the gap between the causal score and the same classifier's score with 16.5 s of
lookahead.

Three overnight recordings: **DH3, AO0, MF7**. Two zygomatic channels, **E227** and **E254**,
1000 Hz. Evaluation is **leave-one-recording-out**, **exact-epoch matching**, scored **from
second 1200**, at threshold **0.70** with a **2 s refractory** period.

The manuscript's numbers already exist in `dev/RESULTS.md`. Your job is to produce the
underlying arrays and count tables that those summaries were computed from, so that
distributions and curves can be plotted rather than just point values.

### Where the code is

**The research code is not in the `zeeg-detector` working tree.** That tree contains only the
release (`detect.ipynb`, `constants.py`, `models/`) plus `dev/`. `dev/RESULTS.md` §14 refers to
`latency.py`, `latency.log`, the earlier `realtime_fir101.py` simulator and a `cache/` directory that is
`.gitignore`d and absent.

**First task: locate the research code and the `cache/` directory.** Ask the user where it
lives if it is not obvious. Do not reimplement anything — if you cannot find the original
code, stop and report that, rather than writing a substitute.

---

## 1. Ground rules

1. **Do not retrain either model.** The frozen classifier and the shipped correction
   (`models/xgb_frozen.pkl`, `models/correction.json`) are used as-is.
2. **Do not change hyperparameters, α values, the threshold, or the refractory period**
   except where this brief explicitly asks you to sweep the threshold.
3. **Every exported table must reproduce the published numbers.** §5 lists the checks. If a
   check fails, stop and report the discrepancy — do not "fix" the export to match.
4. **Patient-data hygiene.** Raw signal and per-epoch score time series must not be committed
   to git. Aggregate count tables and latency timings are safe. See §4 for what goes where.
5. If something in this brief contradicts what you find in the code, **the code wins** —
   report the contradiction.

### Fixed configuration

| Setting | Value |
|---|---|
| Epoch | 250 ms |
| Threshold (shipped) | 0.70 |
| Refractory | 2 s |
| Scoring starts | second 1200 |
| Matching | exact epoch |
| Correction depth | 6, all folds |
| α (DH3 / AO0 / MF7) | 1.00 / 0.15 / 0.80 |
| Correction booster | eta 0.05, 300 rounds, min_child_weight 5, reg_lambda 2, max_bin 128, subsample 1.0, colsample_bytree 1.0 |

α is the **leave-one-recording-out** value: when scoring DH3, α = 1.00 and the correction was
fitted on AO0 + MF7 donors, and so on.

---

## 2. Deliverable A — per-window latency (for the latency distribution figure)

**Source:** `latency.log` / whatever `latency.py` writes. If the log holds only summary
statistics, re-run `latency.py` with per-window logging enabled. Re-running is acceptable here
*provided* the medians still reproduce §5.1 — if they drift because the machine is under
different load, report that rather than shipping numbers that disagree with the manuscript.

**Output:** `latency_windows.csv`, one row per timed window, **22,671 rows expected**.

| Column | Meaning | Units |
|---|---|---|
| `recording` | DH3 / AO0 / MF7 | |
| `window_index` | 0-based, within recording | |
| `t_filter_ms` | filter + detrend + normalize | ms |
| `t_features_ms` | feature extraction, 210 features, 2 channels | ms |
| `t_model1_ms` | frozen classifier | ms |
| `t_model2_ms` | correction | ms |
| `t_compute_ms` | sum of the four above | ms |
| `t_end_to_end_ms` | 25.0 + `t_compute_ms` | ms |

Full float precision — do not round. Rounding to the 2 decimals shown in Table 1 destroys the
distribution's shape in the sub-millisecond components.

**Also report in your write-up (not in the CSV):**
- the exact count of windows exceeding 250 ms, and **whether that threshold was applied to
  `t_compute_ms` or to `t_end_to_end_ms`**. RESULTS.md §14 says two windows "exceeded the 250 ms
  epoch ... (max 234 ms)", which only reconciles if 234 ms is the feature-extraction or compute
  maximum and end-to-end then exceeded 250. Confirm which, and give the true maximum of both
  `t_compute_ms` and `t_end_to_end_ms`.
- which recording those windows belong to (RESULTS.md says both are AO0 — verify).

---

## 3. Deliverable B — threshold sweep counts (for the PR curves and the sweep figure)

This is the important one. It must be produced by running **the complete deployable decision
rule** at each threshold, not by post-processing scores in a notebook.

### Model variants

| id | Description |
|---|---|
| `frozen` | Frozen classifier on the causal chain, no correction (equivalently α = 0) |
| `correction_crops` | Correction fitted on the evaluation crops only |
| `correction_nrem_seeddrop` | **Shipped config**: fitted on every donor NREM period ≥ 25 min, each training window's first 20 min dropped, deterministic fit |

All three scored leave-one-recording-out on all three recordings — **9 (variant, recording)
combinations**.

### Sweep

For each combination, sweep the threshold over **0.300 to 0.990 in steps of 0.005** (139
points). At each threshold, apply the full rule in order:

1. threshold the (corrected) probability,
2. apply the 2 s refractory,
3. restrict to the scored span (from second 1200),
4. match against the offline-pipeline reference by **exact epoch**.

**Hold α fixed** at its shipped per-fold value while sweeping. Do **not** re-select α per
threshold — §3.4 of the manuscript shows that jointly tuning α and threshold is exactly the
failure mode being documented, and re-selecting would silently overwrite that result.

**Output:** `threshold_sweep.csv`, 3 × 3 × 139 = **1,251 rows expected**.

| Column | Meaning |
|---|---|
| `variant` | `frozen` / `correction_crops` / `correction_nrem_seeddrop` |
| `recording` | DH3 / AO0 / MF7 |
| `alpha` | α actually used for this fold (0.0 for `frozen`) |
| `threshold` | 0.300 … 0.990 |
| `n_reference` | reference detections in the scored span (constant per recording) |
| `n_detected` | detections emitted after threshold + refractory |
| `tp`, `fp`, `fn` | exact-epoch counts |

Do not export precision/recall/F1 — they are derived, and exporting counts lets the figure code
compute them consistently and lets a reader audit the arithmetic.

---

## 4. Deliverable C — per-epoch scores (optional, local only)

Only if it is cheap. Useful for an example-traces figure later.

`epoch_scores_<RECORDING>.csv`, one row per 250 ms epoch in the scored span:
`epoch_index`, `t_sec`, `p_frozen`, `p_corrected`, `ref_detection` (0/1),
`causal_detection` (0/1, at 0.70 post-refractory), `corrected_detection` (0/1).

**This is a probability time series over a patient's night. Do not commit it to git.** Write it
outside the repo, or to a path already covered by `.gitignore`, and tell the user where it is.

Deliverables A and B are aggregate/timing data and **are** safe to commit.

---

## 5. Verification — run these before reporting done

### 5.1 Latency (Deliverable A), medians / p95 / p99 in ms

| Component | median | p95 | p99 |
|---|---|---|---|
| filter + detrend + normalize | 0.17 | 0.22 | 0.35 |
| feature extraction | 10.17 | 11.00 | 12.8 |
| frozen classifier | 0.40 | 0.54 | 0.72 |
| correction | 0.14 | 0.19 | 0.32 |
| compute | 10.89 | 11.86 | 14.1 |
| end-to-end | 35.9 | 36.9 | 39.1 |

Row count must be 22,671.

### 5.2 Threshold sweep at 0.70 (Deliverable B)

Mean across the three recordings:

| variant | precision | recall | F1 |
|---|---|---|---|
| `frozen` | 0.316 | 0.366 | 0.339 |
| `correction_nrem_seeddrop` | 0.388 | 0.392 | 0.383 |

Per recording, F1 at 0.70:

| recording | `frozen` | `correction_nrem_seeddrop` |
|---|---|---|
| DH3 | 0.334 | 0.426 |
| AO0 | 0.249 | 0.268 |
| MF7 | 0.434 | 0.455 |

`correction_crops` at 0.70 should give mean precision 0.382, mean F1 0.346.

### 5.3 Sweep shape for the shipped variant

Mean precision / recall / F1 for `correction_nrem_seeddrop`:

| threshold | 0.66 | 0.68 | 0.70 | 0.72 | 0.74 | 0.76 | 0.78 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| precision | 0.332 | 0.361 | 0.388 | 0.388 | 0.392 | 0.391 | 0.393 | 0.385 |
| recall | 0.403 | 0.402 | 0.392 | 0.334 | 0.280 | 0.216 | 0.157 | 0.107 |
| F1 | 0.355 | 0.372 | 0.383 | 0.353 | 0.323 | 0.276 | 0.223 | 0.164 |

### 5.4 Best interpolated precision at or above a given recall

| variant | recall ≥ 0.25 | recall ≥ 0.30 | DH3 @ recall 0.45 | MF7 @ recall 0.45 |
|---|---|---|---|---|
| `frozen` | 0.362 | 0.333 | unreachable | 0.455 |
| `correction_crops` | 0.415 | 0.384 | unreachable | 0.514 |
| `correction_nrem_seeddrop` | 0.424 | 0.402 | 0.424 | 0.556 |

"Unreachable" means the frozen classifier never attains recall 0.45 on DH3 at any threshold.
Your sweep must reproduce that — if DH3 `frozen` reaches recall 0.45 somewhere in the grid,
something is wrong with the fold configuration.

### 5.5 Reference counts

The offline pipeline over the full spans yields **261 (DH3), 185 (AO0), 495 (MF7)** detections.
`n_reference` in the sweep is over the *scored span only* (from second 1200) and will be lower;
report the three values you get.

---

## 6. Where to put things

```
<repo>/dev/figures/data/
    latency_windows.csv        # Deliverable A  — safe to commit
    threshold_sweep.csv        # Deliverable B  — safe to commit
    EXPORT_NOTES.md            # your write-up
```

Deliverable C goes outside the repo or under an ignored path.

---

## 7. What to report back

`EXPORT_NOTES.md` should state:

1. Where the research code was found, and the exact commands run.
2. Row counts for each CSV against the expected counts.
3. **Every check in §5, with the value you obtained next to the expected value.** Mark each
   PASS or FAIL. Do not summarize as "all checks passed" — show the numbers.
4. The three `n_reference` values from §5.5.
5. The latency-budget clarification requested at the end of §2 (compute vs end-to-end, and the
   true maxima).
6. Anything in this brief that contradicted the code, and which you followed.
7. Whether `latency.log` was reused or `latency.py` re-run, and if re-run, how the medians
   compare to Table 1.

Do not draw any figures. Export and verify only.
