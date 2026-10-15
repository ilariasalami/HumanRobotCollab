"""Simulated sense component (assignment 3): a virtual user in front of the coach.

Instead of a camera and MediaPipe, SimulatedUser produces the same Features that
Sense.py would report, so the real Think, Policy and exercise trackers run unchanged.

The simulated user has a hidden state (config.USER_STATES) that the robot never sees:
- engaged:     does the exercise at a good pace, stands still between movements
- unmotivated: hesitates, does partial or slow reps, gives up single-leg holds early
- unstable:    sways and leans a lot, rushes reps, has to put the foot down during holds
The state changes at random (rates in config.SIM_RATES). Fatigue, the exercise, the
simplified version and the robot's recent actions change those rates: this is how the
robot's behaviour feeds back into the user, and what makes the loop a collaboration.

Noise in the measurements mimics what we saw in the webcam tests: hip and foot heights
jitter, sway and lean jump to moderate/high for a few frames during normal movement,
and the body is sometimes not detected for a moment.

run_session() runs one full session (sense -> think -> act) and returns one row per
time step; summarize() turns those rows into the numbers used in the evaluation.
"""

import math
import random

from coach import config
from coach.Act import phrase_for
from coach.Belief import BeliefPolicy
from coach.Policy import Action, RulePolicy
from coach.Sense import Features
from coach.Think import Think

ENGAGED, UNMOTIVATED, UNSTABLE = config.USER_STATES

# Robot actions that respond to instability (used by the metrics)
SAFETY_ACTIONS = {'REASSURE_SLOW', 'REASSURE_HOLD', 'SIMPLIFY', 'REST'}

# Time the simulated user needs to stand still before calibration is done (seconds)
CALIBRATION_S = 2.0
# Outside an exercise (instructions, pauses) instability fades at this rate per second
RECOVER_OUTSIDE_EXERCISE = 0.3


class SimulatedUser:
    """A virtual person doing the exercises. Produces Features, reads the screen."""

    def __init__(self, rng, profile='mixed', noise=1.0):
        """:param rng: random.Random, so a session can be repeated with the same seed
        :param profile: key of config.SIM_USERS
        :param noise: multiplier for measurement noise, flicker and dropouts"""
        self.rng = rng
        self.profile = config.SIM_USERS[profile]
        self.noise = noise
        self.seated_drop = rng.uniform(*config.SIM_SEATED_DROP_RANGE)

        # Hidden state (ground truth, never given to the robot)
        self.state = ENGAGED
        self.fatigue = 0.0

        # What the user sees on the screen (updated by see())
        self.flow_state = 'calibrate'
        self.exercise = None
        self.simplified = False
        self.phase = None
        self.action_t = {}              # robot action name -> last time it was given
        self.rest_until = -math.inf

        # Body
        self.drop = 0.0                 # true hip drop (leg lengths)
        self.rep = None                 # current sit-to-stand rep plan
        self.leg = 'left'               # single-leg balance: foot to lift
        self.lift = None                # current lift plan (single-leg balance)
        self.dab_until = -math.inf      # unstable: foot briefly down during a hold
        self.sway = 0.004
        self.lean = 2.0
        self.flicker_until = -math.inf
        self.flicker = (0.0, 0.0)       # extra (sway, lean) during a flicker
        self.dropout_until = -math.inf
        self.last_t = None
        self.dt = 0.0

    # ------------------------------------------------------------------
    # Sense: one time step of the person and the camera
    # ------------------------------------------------------------------
    def step(self, t):
        """Advance the person to time t and return what the camera pipeline would measure.

        :param t: time in seconds
        :return: Features
        """
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        self.dt = dt
        resting = t < self.rest_until
        exercising = self.flow_state == 'doing' and not resting

        self._update_fatigue(dt, exercising)
        self._update_state(t, dt, exercising)

        if self.flow_state == 'doing' and self.exercise == 'sit_to_stand':
            moving = self._sit_to_stand(t, resting)
            lift = 0.0
        elif self.flow_state == 'doing' and self.exercise == 'single_leg_balance':
            self.drop = 0.0
            lift, moving = self._single_leg(t, resting)
        else:
            # Calibration, instructions, pauses: standing still
            self.drop, lift, moving = 0.0, 0.0, False
            self.rep = self.lift = None

        sway, lean = self._balance_signals(t, dt, lift, moving, resting)
        return self._measure(t, lift, sway, lean)

    def _update_fatigue(self, dt, exercising):
        if exercising:
            self.fatigue = min(1.0, self.fatigue + config.SIM_FATIGUE_PER_S * dt)
        else:
            self.fatigue = max(0.0, self.fatigue - config.SIM_RECOVERY_PER_S * dt)

    def _update_state(self, t, dt, exercising):
        """Random change of the hidden state (a continuous-time Markov chain)."""
        if not exercising:
            # Resting or listening to an instruction: instability fades
            if self.state == UNSTABLE and self._happens(RECOVER_OUTSIDE_EXERCISE, dt):
                self._set_state(ENGAGED, t)
            return
        for dest in config.USER_STATES:
            if dest != self.state and self._happens(self._rate(self.state, dest, t), dt):
                self._set_state(dest, t)
                return

    def _rate(self, source, dest, t):
        rate = config.SIM_RATES.get((source, dest), 0.0)
        fatigue = 1.0 + config.SIM_FATIGUE_EFFECT * self.fatigue
        if dest == UNSTABLE:
            rate *= self.profile['unstable'] * config.SIM_EXERCISE_RISK[self.exercise] * fatigue
            if self.simplified:
                rate *= config.SIM_SIMPLIFIED_RISK
        elif dest == UNMOTIVATED:
            rate *= self.profile['unmotivated'] * fatigue
        for (s, d, action), factor in config.SIM_ACTION_EFFECT.items():
            since = t - self.action_t.get(action, -math.inf)
            if (s, d) == (source, dest) and since < config.SIM_ACTION_EFFECT_S:
                rate *= factor
        return rate

    def _set_state(self, state, t):
        self.state = state
        # An unmotivated user who is holding a foot up puts it down soon
        if state == UNMOTIVATED and self.lift and self.lift['down'] == math.inf:
            self.lift['down'] = max(t, self.lift['up']) + self.rng.uniform(1.0, 4.0)
        # Engaged or unstable again: keeps the foot up until the hold is done
        elif state != UNMOTIVATED and self.lift and t < self.lift['down']:
            self.lift['down'] = math.inf

    def _happens(self, rate, dt):
        return self.rng.random() < 1.0 - math.exp(-rate * dt)

    # ------------------------------------------------------------------
    # Body movement
    # ------------------------------------------------------------------
    def _sit_to_stand(self, t, resting):
        """Hip drop over time: lower -> seated (+ hesitation) -> rise -> stand.
        :return: True while the hips are moving"""
        if resting:
            # REST: sit down and stay seated
            self.drop = self.seated_drop
            self.rep = None
            return False
        if self.rep is None or t >= self.rep['end']:
            self.rep = self._plan_rep(t)
        r = self.rep
        if t < r['lowered']:
            u = (t - r['start']) / (r['lowered'] - r['start'])
            self.drop = r['from'] + u * (self.seated_drop - r['from'])
            return True
        if t < r['rise']:
            self.drop = self.seated_drop
            return False
        if t < r['risen']:
            u = (t - r['rise']) / (r['risen'] - r['rise'])
            self.drop = self.seated_drop + u * (r['peak'] - self.seated_drop)
            return True
        self.drop = r['peak']
        return False

    def _plan_rep(self, t):
        """Plan the next rep from the current hidden state."""
        pause, peak = 0.0, 0.0
        if self.state == ENGAGED:
            duration = self.rng.uniform(3.5, 5.5)
        elif self.state == UNSTABLE:
            duration = self.rng.uniform(2.0, 3.8)       # rushes, sometimes too fast
        else:
            choice = self.rng.random()
            if choice < 0.35:                           # hesitates on the chair
                pause = self.rng.uniform(10.5, 14.0)
                duration = self.rng.uniform(5.0, 8.0)
            elif choice < 0.7:                          # does not stand up fully
                duration = self.rng.uniform(5.0, 9.0)
                peak = 0.12
            else:                                       # slow rep
                duration = self.rng.uniform(7.0, 11.0)
        lowered = t + 0.2 * duration
        rise = t + 0.45 * duration + pause
        risen = rise + 0.2 * duration
        return {'start': t, 'from': self.drop, 'lowered': lowered, 'rise': rise,
                'risen': risen, 'end': risen + 0.35 * duration, 'peak': peak}

    def _single_leg(self, t, resting):
        """Foot lift over time, following what the screen shows.
        :return: (signed lift height, moving)"""
        if self.simplified:
            # Both hands on the chair, toe on the floor: no real lift
            return (0.02 if self.leg == 'left' else -0.02), False
        if resting or self.phase not in ('waiting', 'holding'):
            self.lift = None
            return 0.0, False

        if self.lift is None:
            # (delay before lifting, how long the foot stays up, lift height)
            u = self.rng.uniform
            if self.state == ENGAGED:
                up, hold, height = u(1.0, 2.5), math.inf, u(0.10, 0.16)
            elif self.state == UNSTABLE:
                up, hold, height = u(1.0, 2.0), math.inf, u(0.07, 0.12)
            else:
                up, hold, height = u(4.0, 12.0), u(3.0, 8.0), u(0.07, 0.11)
            self.lift = {'up': t + up, 'down': t + up + hold, 'height': height}
        lift = self.lift

        if t >= lift['down'] and self.phase == 'waiting':
            self.lift = None    # the hold was broken: the tracker waits for a new lift
            return 0.0, False
        lifted = lift['up'] <= t < lift['down']
        if lifted and self.state == UNSTABLE:
            # Has to put the foot down for a moment now and then
            if t >= self.dab_until and self._happens(0.4, self.dt):
                self.dab_until = t + self.rng.uniform(0.3, 1.2)
            lifted = t >= self.dab_until
        moving = min(abs(t - lift['up']), abs(t - lift['down'])) < 0.5
        height = lift['height'] if lifted else 0.0
        return (height if self.leg == 'left' else -height), moving

    # ------------------------------------------------------------------
    # Balance signals and measurement
    # ------------------------------------------------------------------
    def _balance_signals(self, t, dt, lift, moving, resting):
        """True sway (shoulder widths above baseline) and trunk lean (degrees)."""
        g = self.rng.gauss
        if self.state == UNSTABLE:
            factor = 0.5 if resting else (0.7 if self.simplified else 1.0)
            sway_target = max(0.0, g(0.05, 0.02)) * factor
            lean_target = abs(g(13.0, 7.0)) * factor
        else:
            sway_target = max(0.0, g(0.004, 0.004))
            lean_target = abs(g(2.0, 1.5))
        if abs(lift) > config.FOOT_LIFT_THRESHOLD:
            sway_target += 0.01
            lean_target += 4.0

        # Short flickers to moderate/high during normal movement
        rate = config.SIM_FLICKER_RATE_MOVING if moving else config.SIM_FLICKER_RATE_STILL
        if t >= self.flicker_until and self._happens(rate * self.noise, dt):
            self.flicker_until = t + self.rng.uniform(*config.SIM_FLICKER_DURATION_S)
            self.flicker = (self.rng.uniform(0.02, 0.07), self.rng.uniform(5.0, 20.0))
        if t < self.flicker_until:
            sway_target += self.flicker[0]
            lean_target += self.flicker[1]

        # Sway is measured over a time window in Sense, so it changes smoothly
        self.sway += 0.3 * (sway_target - self.sway)
        self.lean += 0.3 * (lean_target - self.lean)
        return self.sway, self.lean

    def _measure(self, t, lift, sway, lean):
        """Features as Sense would report them, with measurement noise and dropouts."""
        calibrated = t >= CALIBRATION_S
        if t >= self.dropout_until and self._happens(config.SIM_DROPOUT_RATE * self.noise, self.dt):
            self.dropout_until = t + self.rng.uniform(*config.SIM_DROPOUT_DURATION_S)
        if t < self.dropout_until:
            return Features(t=t, valid=False, missing=['ankles'], calibrated=calibrated,
                            calib_progress=1.0 if calibrated else t / CALIBRATION_S)

        f = Features(t=t, valid=True, calibrated=calibrated,
                     calib_progress=min(1.0, t / CALIBRATION_S))
        if not calibrated:
            return f
        f.hip_drop = self.drop + self.rng.gauss(0, config.SIM_HIP_DROP_NOISE * self.noise)
        f.foot_lift = lift + self.rng.gauss(0, config.SIM_FOOT_LIFT_NOISE * self.noise)
        if f.foot_lift > config.FOOT_LIFT_THRESHOLD:
            f.lifted_foot = 'left'
        elif f.foot_lift < -config.FOOT_LIFT_THRESHOLD:
            f.lifted_foot = 'right'
        f.sway_baseline = 0.01
        f.sway_excess = max(0.0, sway + self.rng.gauss(0, 0.002 * self.noise))
        f.sway = f.sway_baseline + f.sway_excess
        f.lean_excess = abs(lean + self.rng.gauss(0, 1.0 * self.noise))
        f.trunk_lean = f.lean_excess
        return f

    # ------------------------------------------------------------------
    # The user perceives the robot (what is on screen and what it says)
    # ------------------------------------------------------------------
    def see(self, output, t):
        """:param output: ThinkOutput of this step"""
        if output.state != self.flow_state or output.exercise != self.exercise:
            self.rep = self.lift = None
            if output.exercise != self.exercise:
                self.leg = 'left'
        self.flow_state = output.state
        self.exercise = output.exercise or self.exercise
        self.simplified = output.simplified
        self.phase = output.status.phase if output.status else None
        if 'switch_leg' in output.events:
            self.leg = 'right'
            self.lift = None
        if 'hold_interrupted' in output.events:
            self.lift = None
        if output.action is not Action.NONE:
            self.action_t[output.action.name] = t
            if output.action is Action.REST:
                self.rest_until = t + config.SIM_REST_S


# ----------------------------------------------------------------------
# Running a session
# ----------------------------------------------------------------------
POLICIES = {'rule': RulePolicy, 'hmm': BeliefPolicy}


def run_session(policy='hmm', user='mixed', seed=0, noise=1.0, fps=10.0, max_s=600.0,
                on_step=None):
    """Run one full simulated session: sense -> think -> act, at a fixed time step.

    :param policy: 'rule' or 'hmm'
    :param user: key of config.SIM_USERS
    :param seed: random seed (same seed = same session)
    :param noise: measurement noise multiplier
    :param fps: simulated camera frame rate
    :param max_s: stop after this many simulated seconds if the session is not done
    :param on_step: optional function(row) called after every step (e.g. printing)
    :return: list of rows (dicts), one per time step
    """
    rng = random.Random(seed)
    sim_user = SimulatedUser(rng, user, noise)
    robot_policy = POLICIES[policy]()
    think = Think(policy=robot_policy)
    rows = []
    dt = 1.0 / fps
    step = 0
    done_t = None

    while True:
        t = step * dt
        # Sense (simulated camera)
        features = sim_user.step(t)
        # Think (exercise flow + HMM / rules)
        output = think.step(features, t)
        # Act (the message the robot would show and say)
        message = ''
        if output.action is not Action.NONE:
            message = phrase_for(output.action, output.cue, output.exercise, output.simplified)
        # The user sees the screen and hears the robot
        sim_user.see(output, t)

        belief = getattr(robot_policy, 'belief', None)
        status = output.status
        row = {
            'step': step, 't': round(t, 3),
            'user_state': sim_user.state, 'fatigue': round(sim_user.fatigue, 3),
            'flow_state': output.state, 'exercise': output.exercise or '',
            'simplified': output.simplified,
            'phase': status.phase if status else '',
            'progress': status.progress_text if status else '',
            'valid': features.valid,
            'hip_drop': features.hip_drop, 'foot_lift': features.foot_lift,
            'sway_excess': features.sway_excess, 'lean_excess': features.lean_excess,
            'balance': output.balance or '', 'balance_sustained': output.balance_sustained or '',
            'p_engaged': belief[ENGAGED] if belief else None,
            'p_unmotivated': belief[UNMOTIVATED] if belief else None,
            'p_unstable': belief[UNSTABLE] if belief else None,
            'events': '|'.join(output.events),
            'action': output.action.name if output.action is not Action.NONE else '',
            'message': message,
        }
        rows.append(row)
        if on_step:
            on_step(row)

        if output.state == 'done' and done_t is None:
            done_t = t
        # Keep running a moment after 'done' so the closing message is logged
        if (done_t is not None and t - done_t >= 2.0) or t >= max_s:
            break
        step += 1
    return rows


def summarize(rows):
    """Numbers for comparing policies, computed against the hidden ground truth.

    - unstable_s:         seconds the user was unstable during an exercise
    - unstable_episodes:  number of unstable episodes during an exercise
    - detected:           share of those that got a safety action before they ended
    - latency_s:          mean time from the start of an episode to the first safety action
    - false_alarms:       safety actions while the user was not unstable in the last 2 s
    - motivate_latency_s: mean time from becoming unmotivated to the first MOTIVATE
    - messages:           number of robot messages (how talkative the robot is)
    - belief_accuracy:    share of exercise steps where the most likely HMM state is the
                          true state (HMM policy only)
    """
    doing = [r for r in rows if r['flow_state'] == 'doing']
    dt = rows[1]['t'] - rows[0]['t'] if len(rows) > 1 else 0.0

    def episodes(state):
        result, start = [], None
        for r in rows:
            active = r['user_state'] == state and r['flow_state'] in ('doing', 'demo')
            if active and start is None:
                start = r['t']
            elif not active and start is not None:
                result.append((start, r['t']))
                start = None
        if start is not None:
            result.append((start, rows[-1]['t']))
        return result

    def first_response(spans, actions, grace=0.5):
        latencies = []
        for start, end in spans:
            hit = next((r['t'] for r in rows
                        if start <= r['t'] <= end + grace and r['action'] in actions), None)
            if hit is not None:
                latencies.append(hit - start)
        return latencies

    unstable = episodes(UNSTABLE)
    latencies = first_response(unstable, SAFETY_ACTIONS)
    unmotivated = episodes(UNMOTIVATED)
    motivate_latencies = first_response(unmotivated, {'MOTIVATE'})

    false_alarms = 0
    for i, r in enumerate(rows):
        if r['action'] in SAFETY_ACTIONS:
            window = rows[max(0, i - int(2.0 / dt) if dt else 0):i + 1]
            if not any(w['user_state'] == UNSTABLE for w in window):
                false_alarms += 1

    with_belief = [r for r in doing if r['p_engaged'] is not None]
    accuracy = None
    if with_belief:
        def argmax(r):
            return max(config.USER_STATES, key=lambda s: r[f'p_{s}'])
        accuracy = sum(argmax(r) == r['user_state'] for r in with_belief) / len(with_belief)

    return {
        'duration_s': rows[-1]['t'],
        'completed': rows[-1]['flow_state'] == 'done',
        'unstable_s': sum(dt for r in doing if r['user_state'] == UNSTABLE),
        'unstable_episodes': len(unstable),
        'detected': len(latencies) / len(unstable) if unstable else None,
        'latency_s': sum(latencies) / len(latencies) if latencies else None,
        'false_alarms': false_alarms,
        'unmotivated_episodes': len(unmotivated),
        'motivate_latency_s': (sum(motivate_latencies) / len(motivate_latencies)
                               if motivate_latencies else None),
        'simplified': sum(r['action'] == 'SIMPLIFY' for r in rows),
        'messages': sum(bool(r['action']) for r in rows),
        'belief_accuracy': accuracy,
    }
