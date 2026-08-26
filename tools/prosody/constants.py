HOP = 0.010

PITCH_FLOOR = 60.0
PITCH_CEILING = 400.0

# pass 2 tracks inside a speaker-adapted range; a fixed 60-400 Hz lets low voices track double
ADAPTED_FLOOR_RATIO = 0.55
ADAPTED_CEILING_RATIO = 1.9
ADAPTED_FLOOR = 50.0
ADAPTED_CEILING = 500.0

# a clip whose median falls outside this of its speaker's reference is an octave error
OCTAVE_GUARD = (0.67, 1.5)

MAX_GAP_S = 0.20        # unvoiced gaps longer than this are phrase boundaries, not obstruents
FINAL_S = 0.30          # window used for the phrase-final slope
CONTOUR_POINTS = 64
DCT_COEFFS = 4

MIN_VOICED_FRAMES = 30
MIN_VOICED_FRAC = 0.25
MIN_PHRASE_FRAMES = 10
MIN_DURATION = 2.0      # below this, st_range is inflated by measurement noise

EXPR_EDGES = [10, 35, 65, 90]   # percentiles: the extreme deciles get their own cells
RATE_BINS = 4                   # x 3 phrase-final classes = 60 cells
SLOPE_LEVEL = 5.0               # |st/s| below this counts as level
DEFAULT_SPEAKER_CAP = 2
DEFAULT_FLOOR = 7               # a speaker below this contributes nothing
DEFAULT_CEILING = 20            # and grows toward this only while it flattens the histogram

# absolute Hz, never a pool percentile: a pool is uniform over its own quantiles, so
# quantile edges would reproduce the corpus's pitch skew instead of correcting it
PITCH_EDGE = 165.0
HIGH_FLOOR = 12                 # floor for speakers above the edge, oversampling high voices
