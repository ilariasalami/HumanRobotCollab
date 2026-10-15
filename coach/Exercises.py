from dataclasses import dataclass, field

from coach import config


@dataclass
class ExerciseStatus:
    """What an exercise tracker reports to Think after each frame."""
    exercise: str
    simplified: bool
    phase: str                      # tracker phase, e.g. 'seated', 'holding', 'rest'
    progress_text: str              # short text for the screen, e.g. 'Reps 3/8'
    progress: float                 # 0..1 of the whole exercise
    done: bool
    adherence: float                # 0..1, share of attempts done fully and at a good pace
    events: list = field(default_factory=list)  # events that happened in this frame (see trackers)
    countdown: float = None         # seconds left in the current hold, if holding

    @property
    def adherence_low(self):
        """True in a frame where something happened that means the program is not followed."""
        return any(e in self.events for e in ('partial_rep', 'too_slow', 'inactive', 'hold_broken'))

    @property
    def execution_good(self):
        """True in a frame where a rep/hold was completed correctly."""
        return ('rep_done' in self.events and 'too_fast' not in self.events) or 'hold_done' in self.events


class SitToStand:
    """Counts sit-to-stand reps from the hip height relative to calibration (hip_drop).

    Phases: 'unknown' -> 'seated' -> 'rising' -> 'standing' -> (sits down) 'seated' ...
    A rep counts when the user goes from seated to fully standing.

    Events: 'rep_done', 'partial_rep' (rose past halfway but sat down again),
    'too_fast' / 'too_slow' (time between two reps), 'inactive' (no movement for a while),
    'exercise_done'.
    """

    name = 'sit_to_stand'

    def __init__(self, simplified, t):
        self.simplified = simplified
        self.target = config.STS_TARGET_REPS_SIMPLIFIED if simplified else config.STS_TARGET_REPS
        self.phase = 'unknown'
        self.reps = 0
        self.good_reps = 0
        self.partial_reps = 0
        self.lowest_drop = None         # lowest hip drop reached while rising (closest to standing)
        self.last_rep_t = None
        self.last_activity_t = t
        self.last_t = t

    def update(self, features, t):
        events = []
        if not features.valid or features.hip_drop is None:
            # Tracking lost: pause the timers so this time does not count as inactivity
            self._shift_timers(t - self.last_t)
        else:
            events = self._update_phase(features.hip_drop, t)
        self.last_t = t
        return self._status(events)

    def _update_phase(self, drop, t):
        events = []
        previous_phase = self.phase

        if self.phase == 'unknown':
            if drop < config.STS_STANDING_DROP:
                self.phase = 'standing'
            elif drop > config.STS_SEATED_DROP:
                self.phase = 'seated'

        elif self.phase == 'seated':
            if drop < config.STS_SEATED_DROP:
                self.phase = 'rising'
                self.lowest_drop = drop

        elif self.phase == 'rising':
            self.lowest_drop = min(self.lowest_drop, drop)
            if drop < config.STS_STANDING_DROP:
                self.phase = 'standing'
                events += self._count_rep(t)
            elif drop > config.STS_SEATED_DROP:
                self.phase = 'seated'
                if self.lowest_drop < config.STS_HALFWAY_DROP:
                    self.partial_reps += 1
                    events.append('partial_rep')

        elif self.phase == 'standing':
            if drop > config.STS_SEATED_DROP:
                self.phase = 'seated'

        if self.phase != previous_phase:
            self.last_activity_t = t
        elif t - self.last_activity_t > config.STS_INACTIVE_S and self.reps < self.target:
            events.append('inactive')
            self.last_activity_t = t    # report again only after another full timeout
        return events

    def _count_rep(self, t):
        events = ['rep_done']
        self.reps += 1
        good = True
        if self.last_rep_t is not None:
            interval = t - self.last_rep_t
            if interval < config.STS_MIN_REP_S:
                events.append('too_fast')
                good = False
            elif interval > config.STS_MAX_REP_S:
                events.append('too_slow')
                good = False
        self.last_rep_t = t
        if good:
            self.good_reps += 1
        if self.reps >= self.target:
            events.append('exercise_done')
        return events

    def interrupt(self, t):
        """The robot told the user to rest (Think, on high sway). Nothing to stop, but the
        rest must not count as inactivity or as a slow rep."""
        self.last_activity_t = t
        self.last_rep_t = None
        return []

    def _shift_timers(self, dt):
        self.last_activity_t += dt
        if self.last_rep_t is not None:
            self.last_rep_t += dt

    def _status(self, events):
        attempts = self.reps + self.partial_reps
        return ExerciseStatus(
            exercise=self.name,
            simplified=self.simplified,
            phase=self.phase,
            progress_text=f'Reps {self.reps}/{self.target}',
            progress=min(1.0, self.reps / self.target),
            done=self.reps >= self.target,
            adherence=self.good_reps / attempts if attempts else 1.0,
            events=events,
        )


class SingleLegBalance:
    """Times single-leg holds: SLB_HOLDS_PER_LEG holds of SLB_HOLD_S seconds, first one leg
    then the other.

    Normal version: the timer runs while a foot is lifted (Features.lifted_foot). The first
    leg is whichever foot the user lifts first, the second set needs the other foot.
    Phases: 'waiting' -> 'holding' -> 'rest' -> 'waiting' ...

    Simplified version (both hands on the chair, toe may stay on the floor): the lift is not
    detected. Phases: 'get_ready' -> 'holding' (countdown) -> 'rest' -> 'get_ready' ...

    Events: 'hold_started', 'hold_done', 'hold_broken' (foot down too early),
    'hold_interrupted' (stopped by Think, e.g. for safety), 'switch_leg', 'inactive',
    'exercise_done'.
    """

    name = 'single_leg_balance'

    def __init__(self, simplified, t):
        self.simplified = simplified
        self.leg_set = 1                # 1 = first leg, 2 = other leg
        self.first_leg = None           # 'left'/'right' (normal version only)
        self.holds = 0                  # completed holds on the current leg
        self.hold_start_t = None
        self.last_lift_t = None
        self.phase_until = None         # end of rest / get-ready
        self.waiting_since = t
        self.last_t = t
        self.done = False
        self.completed_holds = 0
        self.broken_holds = 0
        if simplified:
            self._enter('get_ready', t)
        else:
            self.phase = 'waiting'

    def required_leg(self):
        """Which foot must be lifted ('left'/'right'), or None if any foot is fine."""
        if self.simplified or self.first_leg is None:
            return None
        if self.leg_set == 1:
            return self.first_leg
        return 'right' if self.first_leg == 'left' else 'left'

    def interrupt(self, t):
        """The robot told the user to rest (Think, on high sway): stop the current hold
        without counting it and start a rest. Also when no hold is running, otherwise
        the next hold could start right after the robot said to rest."""
        if self.done:
            return []
        events = ['hold_interrupted'] if self.phase == 'holding' else []
        self._enter('rest', t)
        return events

    def update(self, features, t):
        events = []
        if self.done:
            pass
        elif not features.valid:
            # Tracking lost: freeze all timers
            self._shift_timers(t - self.last_t)
        elif self.simplified:
            events = self._update_simplified(t)
        else:
            events = self._update_normal(features.lifted_foot, t)
        self.last_t = t
        return self._status(events, t)

    def _update_normal(self, lifted_foot, t):
        events = []
        required = self.required_leg()
        lifted = lifted_foot is not None and (required is None or lifted_foot == required)

        if self.phase == 'waiting':
            if lifted:
                if self.first_leg is None:
                    self.first_leg = lifted_foot
                self.hold_start_t = t
                self.last_lift_t = t
                self.phase = 'holding'
                events.append('hold_started')
            elif t - self.waiting_since > config.SLB_INACTIVE_S:
                events.append('inactive')
                self.waiting_since = t

        elif self.phase == 'holding':
            if lifted:
                self.last_lift_t = t
            if self.last_lift_t - self.hold_start_t >= config.SLB_HOLD_S:
                events += self._hold_done(t)
            elif t - self.last_lift_t > config.SLB_DROP_GRACE_S:
                self.broken_holds += 1
                events.append('hold_broken')
                self._enter('waiting', t)

        elif self.phase == 'rest':
            if t >= self.phase_until:
                events += self._next_hold_or_leg(t)
        return events

    def _update_simplified(self, t):
        events = []
        if self.phase == 'get_ready':
            if t >= self.phase_until:
                self.hold_start_t = t
                self.phase = 'holding'
                events.append('hold_started')
        elif self.phase == 'holding':
            if t - self.hold_start_t >= config.SLB_HOLD_S:
                events += self._hold_done(t)
        elif self.phase == 'rest':
            if t >= self.phase_until:
                events += self._next_hold_or_leg(t)
        return events

    def _hold_done(self, t):
        self.holds += 1
        self.completed_holds += 1
        self._enter('rest', t)
        return ['hold_done']

    def _next_hold_or_leg(self, t):
        """Called at the end of a rest: start the next hold, switch leg, or finish."""
        events = []
        if self.holds >= config.SLB_HOLDS_PER_LEG:
            if self.leg_set == 1:
                self.leg_set = 2
                self.holds = 0
                events.append('switch_leg')
            else:
                self.done = True
                self.phase = 'done'
                return ['exercise_done']
        self._enter('get_ready' if self.simplified else 'waiting', t)
        return events

    def _enter(self, phase, t):
        self.phase = phase
        if phase == 'rest':
            self.phase_until = t + config.SLB_REST_S
        elif phase == 'get_ready':
            self.phase_until = t + config.SLB_GET_READY_S
        elif phase == 'waiting':
            self.waiting_since = t

    def _shift_timers(self, dt):
        for name in ('hold_start_t', 'last_lift_t', 'phase_until'):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value + dt)
        self.waiting_since += dt

    def _status(self, events, t):
        countdown = None
        if self.phase == 'holding':
            reference = t if self.simplified else self.last_lift_t
            countdown = max(0.0, config.SLB_HOLD_S - (reference - self.hold_start_t))

        total_holds = 2 * config.SLB_HOLDS_PER_LEG
        done_holds = (self.leg_set - 1) * config.SLB_HOLDS_PER_LEG + self.holds
        if self.done:
            text = 'Done'
        else:
            leg = self.required_leg()
            leg_text = f'{leg} leg' if leg else f'leg {self.leg_set}'
            text = f'{leg_text}: hold {min(self.holds + 1, config.SLB_HOLDS_PER_LEG)}/{config.SLB_HOLDS_PER_LEG}'

        attempts = self.completed_holds + self.broken_holds
        return ExerciseStatus(
            exercise=self.name,
            simplified=self.simplified,
            phase=self.phase,
            progress_text=text,
            progress=min(1.0, done_holds / total_holds),
            done=self.done,
            adherence=self.completed_holds / attempts if attempts else 1.0,
            events=events,
            countdown=countdown,
        )
