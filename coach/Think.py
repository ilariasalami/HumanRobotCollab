"""Think component: decides what the robot does. No camera code and no drawing here.

Three parts:
- balance_level() / BalanceFilter: turn the sway and lean numbers from Sense into a
  discrete balance level that has lasted long enough to react to.
- The exercise-flow state machine (transitions library):
  calibrate -> demo -> doing -> (demo of the next exercise ...) -> done.
  The exercise and its version (normal / simplified) are attributes, not states.
- Each frame while 'doing', a small discrete Observation goes to the policy
  (Policy.RulePolicy), which answers with one Action. Think applies cooldowns and
  carries out the effect of the action on the flow (e.g. switching to the simplified
  version).
"""

from dataclasses import dataclass, field

from transitions import Machine

from coach import config
from coach.Exercises import SitToStand, SingleLegBalance
from coach.Policy import Action, Observation, PRIORITY, RulePolicy


LOW, MODERATE, HIGH = 'low', 'moderate', 'high'

# Tracker class for each name in config.EXERCISE_ORDER
EXERCISES = {cls.name: cls for cls in (SitToStand, SingleLegBalance)}

# Events that mean the program is not followed, in the order used to pick the
# robot's wording when several happen in the same frame.
LOW_ADHERENCE_EVENTS = ('partial_rep', 'hold_broken', 'inactive', 'too_slow')

# Longest time step counted by the BalanceFilter, so a pause in the camera stream
# does not count as seconds of instability.
MAX_FRAME_GAP_S = 0.25


def balance_level(features):
    """Discretise balance into 'low' / 'moderate' / 'high' (amount of instability).

    The level is the worst of two signals, both relative to calibration:
    sway above baseline and sideways trunk lean. Returns None when not available
    (not calibrated or body not visible).
    """
    if not features.valid or features.sway_excess is None or features.lean_excess is None:
        return None

    def level(value, moderate, high):
        if value >= high:
            return HIGH
        if value >= moderate:
            return MODERATE
        return LOW

    levels = [
        level(features.sway_excess, config.SWAY_MODERATE, config.SWAY_HIGH),
        level(features.lean_excess, config.LEAN_MODERATE_DEG, config.LEAN_HIGH_DEG),
    ]
    for candidate in (HIGH, MODERATE):
        if candidate in levels:
            return candidate
    return LOW


class BalanceFilter:
    """Reports a balance level only after it has lasted (config.BALANCE_*_SUSTAIN_S).

    Frames without a level (None, e.g. just after a foot is lifted) are ignored: they
    neither count as unstable nor reset the timers.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.moderate_s = 0.0   # how long the level has been moderate or worse
        self.high_s = 0.0       # how long the level has been high
        self.last_t = None

    def update(self, level, t):
        """:param level: balance_level() of this frame
        :return: the sustained level: 'low', 'moderate' or 'high'"""
        dt = 0.0 if self.last_t is None else min(t - self.last_t, MAX_FRAME_GAP_S)
        self.last_t = t
        if level == HIGH:
            self.moderate_s += dt
            self.high_s += dt
        elif level == MODERATE:
            self.moderate_s += dt
            self.high_s = 0.0
        elif level == LOW:
            self.moderate_s = 0.0
            self.high_s = 0.0

        if self.high_s >= config.BALANCE_HIGH_SUSTAIN_S:
            return HIGH
        if self.moderate_s >= config.BALANCE_MODERATE_SUSTAIN_S:
            return MODERATE
        return LOW


@dataclass
class ThinkOutput:
    """Everything Think decided in one frame. Act draws it, main logs it."""
    state: str                      # flow state: 'calibrate', 'demo', 'doing', 'done'
    exercise: str = None            # current exercise name (None in 'calibrate' and 'done')
    simplified: bool = False
    status: object = None           # ExerciseStatus of the tracker while 'doing'
    balance: str = None             # balance level of this frame (can flicker)
    balance_sustained: str = None   # balance level that has lasted, the one the policy sees
    action: Action = Action.NONE    # robot action that starts in this frame
    cue: str = None                 # what the action is about, used by Act to pick the words
    events: list = field(default_factory=list)  # tracker events and state changes, for logging


class Think:
    """Exercise-flow state machine plus the call to the policy."""

    STATES = ['calibrate', 'demo', 'doing', 'done']

    def __init__(self, policy=None):
        self.policy = policy or RulePolicy()
        self.exercise_index = 0
        self.simplified = False
        self.tracker = None
        self.demo_until = None
        self.balance_filter = BalanceFilter()
        self.invalid_since = None       # when the body went out of view
        self.pending = []               # flow messages waiting: (not before t, Action, cue)
        self.last_fired = {}            # Action -> time it was last given (cooldowns)
        self.last_message_t = None
        self.last_message_action = Action.NONE

        # The triggers get a keyword argument t (time) that is passed on to the
        # on_enter_<state> methods below.
        self.machine = Machine(model=self, states=Think.STATES, initial='calibrate', transitions=[
            {'trigger': 'calibrated', 'source': 'calibrate', 'dest': 'demo'},
            {'trigger': 'demo_finished', 'source': 'demo', 'dest': 'doing'},
            {'trigger': 'simplify_exercise', 'source': 'doing', 'dest': 'demo'},
            {'trigger': 'advance', 'source': ['demo', 'doing'], 'dest': 'demo'},
            {'trigger': 'complete', 'source': ['demo', 'doing'], 'dest': 'done'},
            {'trigger': 'restart_calibration', 'source': '*', 'dest': 'calibrate'},
        ])
        self.pending.append((float('-inf'), Action.INSTRUCT, 'calibrate'))

    # ------------------------------------------------------------------
    # State entry (called by the transitions library)
    # ------------------------------------------------------------------
    def on_enter_calibrate(self, t, **kwargs):
        self.tracker = None
        self.pending = [(t, Action.INSTRUCT, 'calibrate')]

    def on_enter_demo(self, t, lead=0.0, announce=True, **kwargs):
        """:param lead: pause before the instruction (after finishing an exercise)
        :param announce: False when another message already explains the exercise"""
        self.tracker = None
        self.balance_filter.reset()
        self.demo_until = t + lead + config.DEMO_S
        self.pending = []
        if announce:
            self.pending.append((t + lead, Action.INSTRUCT, 'demo'))

    def on_enter_doing(self, t, **kwargs):
        self.tracker = EXERCISES[self.exercise_name](self.simplified, t)
        self.balance_filter.reset()
        self.policy.reset()

    def on_enter_done(self, t, lead=0.0, **kwargs):
        self.tracker = None
        self.pending = [(t + lead, Action.INSTRUCT, 'done')]

    # ------------------------------------------------------------------
    # Called from the main loop
    # ------------------------------------------------------------------
    @property
    def exercise_name(self):
        return config.EXERCISE_ORDER[self.exercise_index]

    def recalibrate(self, t):
        """The user asked for a new calibration (key c). The current exercise starts
        again afterwards; after a finished session the whole session starts again."""
        if self.state == 'done':
            self.exercise_index = 0
            self.simplified = False
        self.restart_calibration(t=t)

    def skip_exercise(self, t):
        """Testing aid (key n): go to the next exercise without finishing this one."""
        if self.state in ('demo', 'doing'):
            self._next_exercise(t, lead=0.0)

    def step(self, features, t):
        """One think step.

        :param features: Features from Sense
        :param t: time in seconds
        :return: ThinkOutput
        """
        state_before = self.state
        out = ThinkOutput(state=self.state, balance=balance_level(features))

        # Tracking counts as lost only after the body has been out of view for a while
        if features.valid:
            self.invalid_since = None
        elif self.invalid_since is None:
            self.invalid_since = t
        tracking_ok = self.invalid_since is None or t - self.invalid_since < config.TRACKING_LOST_S

        if self.state == 'calibrate':
            if features.calibrated:
                self.calibrated(t=t)

        elif self.state == 'demo':
            if not tracking_ok:
                self._fire(out, Action.TRACKING_LOST, None, t)
            if t >= self.demo_until:
                self.demo_finished(t=t)

        elif self.state == 'doing':
            self._step_doing(out, features, tracking_ok, t)

        # Flow messages (instructions) wait until nothing else is said in this frame
        if out.action is Action.NONE and self.pending and t >= self.pending[0][0]:
            _, action, cue = self.pending.pop(0)
            self._give(out, action, cue, t)

        out.state = self.state
        if self.state in ('demo', 'doing'):
            out.exercise = self.exercise_name
            out.simplified = self.simplified
        if self.state != state_before:
            out.events.append(f'state:{state_before}->{self.state}')
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _step_doing(self, out, features, tracking_ok, t):
        status = self.tracker.update(features, t)
        out.status = status
        out.events += status.events
        out.balance_sustained = self.balance_filter.update(out.balance, t)

        if 'switch_leg' in status.events:
            self.pending.append((t, Action.INSTRUCT, 'switch_leg'))

        if status.done:
            # Finished: praise now, the next instruction follows after a short pause
            self._give(out, Action.PRAISE, 'exercise_done', t)
            self._next_exercise(t, lead=config.BETWEEN_EXERCISES_S)
            return

        # What the policy is allowed to know
        if 'rep_done' in status.events and 'too_fast' in status.events:
            execution = 'poor'
        elif status.execution_good:
            execution = 'good'
        else:
            execution = None
        observation = Observation(
            t=t,
            tracking_ok=tracking_ok,
            exercise=self.exercise_name,
            simplified=self.simplified,
            balance=out.balance_sustained,
            balance_now=out.balance,
            adherence='low' if status.adherence_low else 'ok',
            execution=execution,
        )
        action = self.policy.decide(observation)

        cue = None
        if action is Action.MOTIVATE:
            cue = next((e for e in LOW_ADHERENCE_EVENTS if e in status.events), None)
        if not self._fire(out, action, cue, t):
            return

        # Effects of the action on the flow
        if action is Action.SIMPLIFY:
            self.simplified = True
            # The SIMPLIFY message itself explains the easier version
            self.simplify_exercise(t=t, announce=False)
        elif action is Action.REST:
            if hasattr(self.tracker, 'interrupt'):
                out.events += self.tracker.interrupt(t)
            self.balance_filter.reset()

    def _next_exercise(self, t, lead):
        if self.exercise_index + 1 < len(config.EXERCISE_ORDER):
            self.exercise_index += 1
            self.simplified = False
            self.advance(t=t, lead=lead)
        else:
            self.complete(t=t, lead=lead)

    def _fire(self, out, action, cue, t):
        """Give a policy action if its cooldown and the minimum gap allow it.
        :return: True if the action was given"""
        if action is Action.NONE:
            return False
        cooldown = config.ACTION_COOLDOWN_S.get(action.name, 0.0)
        if action in self.last_fired and t - self.last_fired[action] < cooldown:
            return False
        if (self.last_message_t is not None
                and t - self.last_message_t < config.ACTION_MIN_GAP_S
                and PRIORITY[action] <= PRIORITY[self.last_message_action]):
            return False
        self._give(out, action, cue, t)
        return True

    def _give(self, out, action, cue, t):
        out.action = action
        out.cue = cue
        self.last_fired[action] = t
        self.last_message_t = t
        self.last_message_action = action
