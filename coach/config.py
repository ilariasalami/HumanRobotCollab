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

# ---------------------------------------------------------------------------
# Think: exercise flow (step 3)
# ---------------------------------------------------------------------------
# The session runs these exercises in this order (names of the trackers in Exercises.py).
EXERCISE_ORDER = ('sit_to_stand', 'single_leg_balance')
# How long the instruction for an exercise is shown before tracking starts.
DEMO_S = 6.0
# Pause after an exercise is finished, so the praise can be read before the next
# instruction replaces it.
BETWEEN_EXERCISES_S = 3.0

# ---------------------------------------------------------------------------
# Think: when does the robot react (step 3)
# ---------------------------------------------------------------------------
# A balance level must last this long before the robot reacts to it. In the webcam
# test the level jumped to moderate/high for a few frames during normal movements
# (sitting down, shifting weight just before lifting a foot); without this the robot
# would reassure or simplify all the time.
BALANCE_MODERATE_SUSTAIN_S = 0.5
BALANCE_HIGH_SUSTAIN_S = 1.0
# The body must be out of view for this long before the robot says it cannot see the user.
TRACKING_LOST_S = 1.5

# The same action is not repeated within its cooldown (seconds), so the robot does not
# say the same thing every frame while a condition lasts. Not listed = no cooldown.
ACTION_COOLDOWN_S = {
    'TRACKING_LOST': 6.0,
    'REST': 10.0,
    'REASSURE_SLOW': 8.0,
    'REASSURE_HOLD': 8.0,
    'MOTIVATE': 8.0,
    'PRAISE': 6.0,
}
# Minimum time between two robot messages. A more urgent action (see Policy.PRIORITY)
# is not held back by this.
ACTION_MIN_GAP_S = 2.5

# ---------------------------------------------------------------------------
# Act: what the robot says (shown as text in step 3, spoken from step 4)
# ---------------------------------------------------------------------------
# How long a robot message stays on screen.
MESSAGE_DISPLAY_S = 6.0

# Key: (action name, context). The context is the cue Think gives with the action
# (e.g. 'demo', 'switch_leg') combined with the exercise; see Act.phrase_for() for
# the order in which keys are tried. (action name, None) is the fallback.
PHRASES = {
    ('INSTRUCT', 'calibrate'): 'Stand still next to your chair, facing the camera.',
    ('INSTRUCT', 'demo:sit_to_stand'):
        'Sit to stand. Sit down, then stand up fully without using your hands, and sit down again. '
        'Do this 8 times.',
    ('INSTRUCT', 'demo:sit_to_stand:simplified'):
        'Sit to stand, easier version. Use your hands on the chair to help you. Do this 5 times.',
    ('INSTRUCT', 'demo:single_leg_balance'):
        'Single-leg balance. Hold the chair with one hand and lift one foot for 10 seconds. '
        '3 times on each leg.',
    ('INSTRUCT', 'demo:single_leg_balance:simplified'):
        'Single-leg balance, easier version. Hold the chair with both hands, you can keep your '
        'toe on the floor. Follow the countdown.',
    ('INSTRUCT', 'switch_leg'): 'Now the other leg.',
    ('INSTRUCT', 'done'): 'That was the whole session. Well done!',

    ('SIMPLIFY', 'sit_to_stand'):
        "Let's make this easier and safer. Use your hands on the chair to help you. 5 times.",
    ('SIMPLIFY', 'single_leg_balance'):
        "Let's make this easier and safer. Hold the chair with both hands, you can keep your "
        'toe on the floor.',
    ('REST', 'sit_to_stand'): 'Sit down and rest for a moment.',
    ('REST', 'single_leg_balance'): 'Put both feet down and rest for a moment.',
    ('REASSURE_SLOW', None): 'Take your time, slow and controlled is better.',
    ('REASSURE_HOLD', None): 'Hold on to the chair, you are doing fine.',
    ('MOTIVATE', None): 'Keep going, you can do this.',
    ('MOTIVATE', 'inactive'): 'Ready when you are. Try the next one.',
    ('MOTIVATE', 'partial_rep'): 'Almost. Try to stand up all the way.',
    ('MOTIVATE', 'hold_broken'): 'Good try. Lift your foot again and hold a bit longer.',
    ('PRAISE', None): 'Well done!',
    ('PRAISE', 'exercise_done'): 'Exercise finished, great work!',
    ('TRACKING_LOST', None): 'I cannot see your whole body. Please step back into view.',

    # The robot's own wobble (coach/Collaboration.py)
    ('ROBOT_WOBBLE', None): "Whoa, I'm a bit wobbly myself! Can you cheer me on?",
    ('THANK', None): 'Thank you, that helped! We make a good team.',
    ('ROBOT_STEADY', None): "Okay, I'm steady again. Let's keep going.",
}

# ---------------------------------------------------------------------------
# Think: HMM belief over the user's state (assignment 3, coach/Belief.py)
# ---------------------------------------------------------------------------
# Hidden user states the robot reasons about:
#   engaged      follows the program, stable
#   unmotivated  the program is not followed (slow, partial reps, gives up holds)
#   unstable     needs support: sways or leans clearly more than when calibrated
USER_STATES = ('engaged', 'unmotivated', 'unstable')

# The HMM is updated at a fixed time step instead of every frame, so the same model
# works at 10, 15 or 30 frames per second. Within a step the most frequent balance
# level counts as the observation (a single flickering frame does not decide).
HMM_STEP_S = 0.25

# Belief at the start of the session: most people start engaged and stable.
HMM_PRIOR = {'engaged': 0.8, 'unmotivated': 0.1, 'unstable': 0.1}

# Transition rates per second (from -> to). Converted to probabilities per step with
# p = 1 - exp(-rate * HMM_STEP_S). Small rates = states last for many seconds, which
# is what keeps the belief from following every flicker of the balance level.
HMM_RATES = {
    ('engaged', 'unmotivated'): 0.02,
    ('engaged', 'unstable'): 0.03,
    ('unmotivated', 'engaged'): 0.05,
    ('unmotivated', 'unstable'): 0.01,
    ('unstable', 'engaged'): 0.10,
    ('unstable', 'unmotivated'): 0.02,
}

# P(balance level of a step | user state). Balance does not tell engaged and
# unmotivated apart; both show short flickers to moderate/high during normal movement
# (seen in the webcam tests), unstable shows them most of the time.
HMM_BALANCE_EMISSION = {
    'engaged':     {'low': 0.86, 'moderate': 0.11, 'high': 0.03},
    'unmotivated': {'low': 0.86, 'moderate': 0.11, 'high': 0.03},
    'unstable':    {'low': 0.20, 'moderate': 0.40, 'high': 0.40},
}

# Relative likelihood of a tracker event given the state. Only used in a step where
# such an event happens: no event is treated as no evidence, because how often events
# happen depends on the exercise (a rep every ~4 s, a hold every ~13 s). Long
# inactivity is reported by the trackers as its own 'inactive' event.
#   low_adherence = partial_rep / too_slow / inactive / hold_broken
#   too_fast      = rep done in less than STS_MIN_REP_S
#   good          = rep or hold done correctly
#   encouraged    = the user encouraged the wobbling robot (an engaged user does this most)
HMM_EVENT_LIKELIHOOD = {
    'engaged':     {'good': 1.0, 'low_adherence': 0.1, 'too_fast': 0.2, 'encouraged': 1.0},
    'unmotivated': {'good': 0.3, 'low_adherence': 1.0, 'too_fast': 0.1, 'encouraged': 0.4},
    'unstable':    {'good': 0.4, 'low_adherence': 0.5, 'too_fast': 1.0, 'encouraged': 0.2},
}

# At the start of each exercise the belief is reset to a mix of HMM_PRIOR and the
# average belief of the session so far. This is how the robot adapts to the user: a
# user who was unstable before starts the next exercise with a higher P(unstable).
HMM_SESSION_MEMORY = 0.5

# BeliefPolicy: belief thresholds for the robot actions (same priority order as the
# rule policy). The SIMPLIFY threshold is high because it changes the exercise.
BELIEF_SIMPLIFY_P = 0.9
BELIEF_REASSURE_P = 0.6
BELIEF_MOTIVATE_P = 0.5

# ---------------------------------------------------------------------------
# Collaboration: the robot's own wobble (assignment 3, coach/Collaboration.py)
# ---------------------------------------------------------------------------
# The robot exercises along with the user and loses its balance a little once or twice
# per session, so the user can support the robot too (support goes both ways).
ROBOT_WOBBLES = (1, 2)                  # number of wobbles per session (min, max)
# When: after this much exercise time (seconds in 'doing') since the start of the session.
# A session has roughly 130-200 s of exercise time.
ROBOT_WOBBLE_WINDOW_S = (15.0, 110.0)
ROBOT_WOBBLE_MIN_GAP_S = 25.0           # exercise time between two wobbles
ROBOT_WOBBLE_MAX_S = 8.0                # steadies itself after this long if nobody helps
# Safety first: no wobble within this time after a safety action, or while the user's
# balance level is moderate/high.
ROBOT_WOBBLE_SAFE_GAP_S = 5.0

# ---------------------------------------------------------------------------
# Simulation (assignment 3, coach/Simulation.py): the simulated user
# ---------------------------------------------------------------------------
# None of this is used by the robot. These numbers make the simulated person behave
# roughly like we did in the webcam tests; they are deliberately NOT the same as the
# HMM parameters above (the robot's model of a user is never exact).

# Transition rates of the simulated user's hidden state per second, while exercising.
SIM_RATES = {
    ('engaged', 'unmotivated'): 0.010,
    ('engaged', 'unstable'): 0.010,     # multiplied by SIM_EXERCISE_RISK and fatigue
    ('unmotivated', 'engaged'): 0.03,   # multiplied by SIM_ACTION_EFFECT after MOTIVATE
    ('unmotivated', 'unstable'): 0.005,
    ('unstable', 'engaged'): 0.06,      # multiplied by SIM_ACTION_EFFECT after a safety action
    ('unstable', 'unmotivated'): 0.02,
}
# Single-leg balance is harder than sit-to-stand; the simplified versions are easier.
SIM_EXERCISE_RISK = {'sit_to_stand': 1.0, 'single_leg_balance': 2.5}
SIM_SIMPLIFIED_RISK = 0.3
# Fatigue (0..1) grows while exercising and recovers while resting. It multiplies the
# chance of becoming unstable or unmotivated by (1 + SIM_FATIGUE_EFFECT * fatigue).
SIM_FATIGUE_PER_S = 0.006
SIM_RECOVERY_PER_S = 0.03
SIM_FATIGUE_EFFECT = 3.0
# How the robot's actions change the user's transition rates, for SIM_ACTION_EFFECT_S
# seconds after the action: (from state, to state, robot action) -> multiplier.
SIM_ACTION_EFFECT_S = 6.0
SIM_ACTION_EFFECT = {
    ('unmotivated', 'engaged', 'MOTIVATE'): 5.0,
    ('unmotivated', 'engaged', 'PRAISE'): 1.5,
    ('unstable', 'engaged', 'REASSURE_SLOW'): 2.5,
    ('unstable', 'engaged', 'REASSURE_HOLD'): 2.5,
    ('unstable', 'engaged', 'SIMPLIFY'): 6.0,
    ('unstable', 'engaged', 'REST'): 6.0,
    # Being thanked by the robot for helping it gives a small motivation boost
    ('unmotivated', 'engaged', 'THANK'): 3.0,
}
# Chance that the simulated user encourages the wobbling robot, by hidden state: an
# unstable user is busy with their own balance. Encouragement starts after a reaction
# time and lasts (is heard) for a moment.
SIM_ENCOURAGE_P = {'engaged': 0.9, 'unmotivated': 0.5, 'unstable': 0.1}
SIM_ENCOURAGE_DELAY_S = (1.0, 3.0)
SIM_ENCOURAGE_DURATION_S = 1.5
# After REST the simulated user sits / puts both feet down for this long.
SIM_REST_S = 5.0
# A state lasts at least this long before it can change again: a person's motivation or
# balance does not switch back and forth every second.
SIM_MIN_STATE_S = 3.0

# User profiles: multipliers on the base rates (simulate.py --user).
SIM_USERS = {
    'steady':      {'unstable': 0.3, 'unmotivated': 0.3},
    'mixed':       {'unstable': 1.0, 'unmotivated': 1.0},
    'shaky':       {'unstable': 3.0, 'unmotivated': 0.7},
    'unmotivated': {'unstable': 0.7, 'unmotivated': 3.0},
}

# Sensor model: what the camera pipeline would report, with noise.
SIM_SEATED_DROP_RANGE = (0.26, 0.5)     # per user, both values were measured in our tests
SIM_HIP_DROP_NOISE = 0.01
SIM_FOOT_LIFT_NOISE = 0.015
# Short bursts where sway/lean jump to moderate/high during normal movement (as in the
# webcam tests). Rate per second, during movement and at rest; scaled by --noise.
SIM_FLICKER_RATE_MOVING = 0.35
SIM_FLICKER_RATE_STILL = 0.05
SIM_FLICKER_DURATION_S = (0.15, 0.6)
# Short detection glitches: the body is not detected for a moment although the user did
# not move (rate per second, scaled by --noise). Shorter than TRACKING_LOST_S, so the
# robot correctly does not react to them.
SIM_DROPOUT_RATE = 0.01
SIM_DROPOUT_DURATION_S = (0.2, 1.0)
# The user walks up to the laptop (e.g. to read the screen): their feet leave the camera
# view, the robot asks them to step back and they walk back. Rate per second during
# instructions and exercises: on average about once every two sessions.
SIM_WALK_AWAY_RATE = 0.004
SIM_WALK_S = 1.5                        # walking time each way
SIM_RETURN_DELAY_S = (1.0, 2.0)         # reaction time after the robot asks
SIM_AWAY_MAX_S = 10.0                   # walks back by itself if the robot says nothing
