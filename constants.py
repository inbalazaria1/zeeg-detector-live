"""Every constant the detector uses, in one place.

`detect.ipynb` and `dev/train.ipynb` both `from constants import *`, so the two
notebooks can't drift apart.

Nothing here is a tuning knob (setting you're allowed to change to adjust how the system behaves).
Each value is a property of the recording format,
a unit conversion, or what the frozen model was trained at. Change one and the
numbers in `README.md` stop applying.

Left out on purpose: the feature parameters (`emb=2`, `q=0.75`, `kmax=10`, the
band edges), which stay inline in `extract_epoch_features` because that code was
copied byte-for-byte from the training code; and the seeds and dummy values in
the self-checks, which mean nothing and only have to stay put.
"""

# ======================================================================
# units
# ======================================================================
SECONDS_PER_MINUTE = 60
MILLISECONDS_PER_SECOND = 1000
MICROVOLTS_TO_VOLTS = 1e-6  # what MNE applies to raw EGI samples

# ======================================================================
# the recording
# ======================================================================
SAMPLING_RATE_HZ = 1000.0             # Hz - the only rate the model was trained at
ANALYSIS_CHANNELS = ["E227", "E254"]  # the two analysis channels, hard-wired
N_MODEL_FEATURES = 210                # 105 per channel; the frozen model's input width
CHANNEL_PREFIX_LEN = 5                # len("eog1_") - how feature names are namespaced

# ======================================================================
# epochs - one decision per epoch
# ======================================================================
EPOCH_SAMPLES = 250                         # samples in one decision window
EPOCH_SECONDS = 0.25
EPOCHS_PER_SECOND = int(1 / EPOCH_SECONDS)  # 4 - the refractory is counted in these
EPOCHS_PER_MINUTE = 240                     # 60 s / 0.25 s
# Four epochs = the last second.  The offline code we copied stopped a second
# early, so we do too, or the epoch numbers stop matching the reference.
DROPPED_TAIL_EPOCHS = 4

# ======================================================================
# stage 1 - band-pass.  LOOKAHEAD_SAMPLES is the whole 25 ms of lookahead.
# ======================================================================
# Not free choices: both models were fitted on features from this exact filter,
# so changing either means refitting.  The tap count also sets the lookahead -
# fewer taps means less delay and a softer 70 Hz cut.
BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ = 0.1, 70.0  # band edges, Hz
HIGHPASS_ORDER = 3                             # causal Butterworth order for the 0.1 Hz corner
FIR_TAP_COUNT = 51                             # symmetric 70 Hz firwin
LOOKAHEAD_SAMPLES = (FIR_TAP_COUNT - 1) // 2   # read 25 samples late
LOOKAHEAD_MS = LOOKAHEAD_SAMPLES / SAMPLING_RATE_HZ * MILLISECONDS_PER_SECOND

# ======================================================================
# stage 3 - detrend
# ======================================================================
DETREND_WINDOW_SECONDS = 2.0  # trailing box window

# ======================================================================
# stage 6 - normalisation
# ======================================================================
# Structural, not a default to lower.  The expanding z-score needs this much of
# a patient, and it is what everything was fitted at.  Nothing fires inside it.
NORMALISATION_MINUTES = 20.0
NORMALISATION_SECONDS = NORMALISATION_MINUTES * SECONDS_PER_MINUTE
FILTER_WARMUP_MINUTES = 1.0  # filter warm-up read before the analysed span
N_RAW_MOMENTS = 4            # sums accumulated: x, x^2, x^3, x^4
VARIANCE_FLOOR = 1e-30       # guards a divide-by-zero on a flat channel

# ======================================================================
# the two models
# ======================================================================
MODEL_DIR = "models"            # relative to the repository root
PROBABILITY_CLIP_EPS = 1e-6     # clip before logit(), both directions
WARMUP_MATCH_TOLERANCE = 1e-12  # the fast gather must equal the DataFrame path

# ======================================================================
# stage 7 - threshold, refractory, bins
# ======================================================================
# Four rules that move it off 0.70 were tried; all four did worse, and the
# shipped correction was picked at 0.70.  See dev/RESULTS.md §2 and §6.
DETECTION_THRESHOLD = 0.70
REFRACTORY_SECONDS = 2   # refractory, seconds
BIN_LENGTH_MINUTES = 20  # detection bin, minutes
EPOCHS_PER_BIN = int(BIN_LENGTH_MINUTES * EPOCHS_PER_MINUTE)

# ======================================================================
# .mff binary layout
# ======================================================================
# An MFF signal file is a run of blocks.  Only the first has a header
# (version == 1); later ones open with a 4-byte zero meaning "same geometry as
# before", so block k's offset is arithmetic.
BYTES_PER_SAMPLE = 4            # int32-as-float32 samples
MFF_HEADER_BYTES = 16           # version, header_size, block_size, n_channels
MFF_MARKER_BYTES = 4            # the version==0 word every later block opens with
MFF_BIT_DEPTH = 32              # the only depth this reader handles
MFF_DEPTH_MASK = 0xFF           # low byte of the sigfreq word is the depth
MFF_RATE_SHIFT_BITS = 8         # ...and the rest of it, shifted, is the rate
MFF_VREF_INDEX = 256            # 0-based; VREF/Cz is EGI channel 257
MFF_MARKER_CHECK_STRIDE = 32    # verify one block marker in every N touched
IO_CHUNK_SECONDS = 4.0          # how much the reader pulls per I/O
SAMPLING_RATE_TOLERANCE = 1e-9  # tolerance when checking the file's rate

# ======================================================================
# reporting - formatting, except TIME_DECIMALS (see below)
# ======================================================================
DIVISION_GUARD = 1e-9           # denominator floor in rate/throughput prints
BYTES_PER_KB = 1024
# Decimal places on the detections CSV.  TIME_DECIMALS is not just display:
# dev/train.ipynb rounds detected and reference times to it before matching
# them, so changing it changes the scores.  Detections fall on multiples of
# 0.25 s, so 2 places is exact; minutes need 4 to stay readable.
TIME_DECIMALS = 2               # time_sec
MINUTE_DECIMALS = 4             # time_min
PROBABILITY_DECIMALS = 6        # probability
LATENCY_PERCENTILE = 95         # the p95 column of the latency table
PROGRESS_EVERY_N_EPOCHS = 8000  # epochs between progress lines
PREVIEW_ROWS = 20               # rows of the detection table shown inline

# ======================================================================
# model 2, the training recipe.  Read by dev/train.ipynb only.
# ======================================================================
# Every NREM period at least this long is a training window.  The first 20 min
# are the seed, so 25 leaves about 5 min of rows.  Whether a window can also be
# scored is a separate test - see MIN_VALIDATION_REF_SPIKES.
MIN_TRAINING_WINDOW_MINUTES = 25
# A validation window needs something to score.  Stage 7 bins in 20 min and
# drops a final bin under 10, and the first 20 min are the seed, so a 29 min
# period scores nothing.
MIN_VALIDATION_REF_SPIKES = 20

XGB_ETA = 0.05
XGB_ROUNDS = 300
XGB_DEPTHS = (3, 6)  # the only hyper-parameter searched, besides alpha
XGB_MIN_CHILD_WEIGHT = 5
XGB_REG_LAMBDA = 2.0
# 1.0 / 1.0 makes the fit deterministic: nothing random, the seed is inert, and
# two fits produce byte-identical model files.
XGB_SUBSAMPLE = 1.0
XGB_COLSAMPLE = 1.0
XGB_MAX_BIN = 128
# alpha is not an XGBoost setting.  It is how much of model 2's correction is
# added, in logit space: logit(frozen) + alpha * correction.  0 ignores the
# correction, 1 takes all of it.  Chosen with the depth, leave-one-recording-out.
ALPHA_GRID = (0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0)
DEFAULT_N_THREADS = 8
