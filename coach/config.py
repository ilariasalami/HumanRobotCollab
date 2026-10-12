# All tunable numbers of the coach in one place.
# None of these values are clinically validated: they are starting points that we tune
# on ourselves. Distances are normalised (see comments) so they do not depend on how far
# the user stands from the camera.

# ---------------------------------------------------------------------------
# Camera / pose detection
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# A landmark counts as visible if MediaPipe's visibility score is above this.
# All of shoulders, hips, knees and ankles must be visible for a frame to be used.
MIN_LANDMARK_VISIBILITY = 0.5

# ---------------------------------------------------------------------------
# Sense: smoothing
# ---------------------------------------------------------------------------
# Exponential moving average factor for angles and heights (0..1).
# Higher = follows the signal faster but noisier.
SMOOTHING_ALPHA = 0.3

# Sway = standard deviation of the hip-midpoint offset over this time window (seconds).
SWAY_WINDOW_S = 1.5
# Minimum number of samples in the window before a sway value is reported.
SWAY_MIN_SAMPLES = 10

# ---------------------------------------------------------------------------
# Sense: calibration (user stands still next to the chair, facing the camera)
# ---------------------------------------------------------------------------
# First the user gets time to settle (walking back from the laptop etc.), these samples
# are thrown away. Equal to the sway window, so the first sway values measured are
# not polluted by the settling movement.
CALIBRATION_SETTLE_S = 1.5
# Then this long the user has to stand still to measure the baselines.
CALIBRATION_DURATION_S = 3.0
# Calibration restarts if the hip midpoint moves more than this (in shoulder widths)
# away from where it was at the start of the measurement (~0.08 = about 3 cm).
CALIBRATION_STILL_TOLERANCE = 0.08
# Upper bound for the baseline sway (in shoulder widths), so that a messy calibration
# cannot hide real wobbling. Standing still measured 0.006-0.02 in our first tests.
MAX_BASELINE_SWAY = 0.03

# ---------------------------------------------------------------------------
# Sense: foot lift (used to choose the reference point for sway)
# ---------------------------------------------------------------------------
# Ankle height difference (in leg lengths, relative to calibration) above which
# one foot counts as lifted.
FOOT_LIFT_THRESHOLD = 0.06

# ---------------------------------------------------------------------------
# Balance levels
# ---------------------------------------------------------------------------
# Excess sway = current sway - calibrated sway, in shoulder widths (1 shoulder width is
# roughly 35-40 cm, so 0.025 is about 1 cm standard deviation of the hip position).
# We subtract instead of divide: the baseline is mostly camera jitter and dividing by a
# small noisy number made the levels jump between calibrations.
# First test: standing still ~0, lifting a foot ~0.01, almost falling ~0.12.
SWAY_MODERATE = 0.025   # above this: reassure ("slow down" / "hold the chair")
SWAY_HIGH = 0.06        # above this: switch to the simplified version

# Sideways trunk lean compared to the calibrated lean, in degrees. Counts as a second
# balance signal: the balance level is the worst of sway and lean.
# First test: standing ~1-2 deg, lifting a foot ~5-8 deg, almost falling ~34 deg.
LEAN_MODERATE_DEG = 12
LEAN_HIGH_DEG = 25

# ---------------------------------------------------------------------------
# Exercise: sit-to-stand (next to / in front of a chair)
# ---------------------------------------------------------------------------
# Hip drop = how far the hips are below their calibrated standing height, in leg
# lengths. First test: standing ~0.0, seated ~0.5.
# Second tester (other chair and camera position): standing 0.00 +- 0.03, seated only
# ~0.26, a half rise reached ~0.13. With the first values (0.10 / 0.35 / 0.22) no rep
# was counted for her, so the thresholds are lowered to fit both testers. The seated
# depth clearly differs per person and setup: measuring it per user would be better.
STS_STANDING_DROP = 0.07   # below this the user is standing (full stand reached)
STS_SEATED_DROP = 0.20     # above this the user is seated
STS_HALFWAY_DROP = 0.16    # rising above this but sitting down again = partial rep

STS_TARGET_REPS = 8
STS_TARGET_REPS_SIMPLIFIED = 5

# Pace, measured per rep from leaving the chair to leaving it again.
STS_MIN_REP_S = 2.5        # faster than this: too fast (less controlled)
STS_MAX_REP_S = 12.0       # slower than this: low adherence
STS_INACTIVE_S = 10.0      # no movement for this long: low adherence

# ---------------------------------------------------------------------------
# Exercise: single-leg balance (holding the chair)
# ---------------------------------------------------------------------------
# Normal version: the hold timer runs while one foot is lifted (Sense FOOT_LIFT_THRESHOLD).
# Simplified version (both hands on the chair, toe may touch the floor): no lift
# detection, the robot gives a short get-ready time and then counts down.
SLB_HOLD_S = 10.0          # target hold time
SLB_HOLDS_PER_LEG = 3
# A lift detection gap shorter than this does not break the hold. Was 0.4: with a low
# lift (foot ~0.05-0.13 leg lengths up) detection dropped out for 0.6 s while the foot
# was still in the air, which broke the hold.
SLB_DROP_GRACE_S = 0.8
SLB_REST_S = 3.0           # rest between holds (both feet down)
SLB_GET_READY_S = 4.0      # simplified version: time to get ready before the countdown
SLB_INACTIVE_S = 10.0      # normal version: no lift for this long: low adherence

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_NAME = 'Osteoporosis Coach'
# Show the camera image mirrored, so it feels like looking in a mirror.
MIRROR_DISPLAY = True
