"""Intent recognition: a hidden Markov model of the user's state, and a policy that acts on it.

The robot cannot see whether the user is engaged, unmotivated or unstable. It only sees
noisy signals: a balance level that flickers during normal movement, and tracker events
such as a partial rep or a broken hold. UserStateHMM keeps a belief (a probability for
each state) and updates it with the forward algorithm:

    predict:  b'(s') = sum_s b(s) * A(s -> s')
    correct:  b(s')  = b'(s') * P(observation | s') / normalisation

BeliefPolicy has the same interface as Policy.RulePolicy (reset() and decide()), so
Think does not need to know which one it uses. It keeps the rule policy's priority
(tracking > safety > adherence > praise) but decides on the belief instead of on a
sustained balance level or a single event.

All numbers are in config.py (HMM_* and BELIEF_*).
"""

import math
from collections import Counter

from coach import config
from coach.Policy import Action

STATES = config.USER_STATES

# Observation fields -> event symbols of the HMM (see config.HMM_EVENT_LIKELIHOOD)
LOW_ADHERENCE = 'low_adherence'
TOO_FAST = 'too_fast'
GOOD = 'good'


def transition_matrix(rates, step_s):
    """Turn transition rates per second into probabilities per step.

    :param rates: dict (from, to) -> rate per second
    :param step_s: step length in seconds
    :return: dict from -> dict to -> probability (each row sums to 1)
    """
    matrix = {}
    for source in STATES:
        row = {}
        for dest in STATES:
            if dest != source:
                row[dest] = 1.0 - math.exp(-rates.get((source, dest), 0.0) * step_s)
        row[source] = 1.0 - sum(row.values())
        matrix[source] = row
    return matrix


class UserStateHMM:
    """Forward filter over config.USER_STATES, updated every config.HMM_STEP_S seconds."""

    def __init__(self):
        self.A = transition_matrix(config.HMM_RATES, config.HMM_STEP_S)
        self.belief = dict(config.HMM_PRIOR)
        # Running average of the belief over the session (for reset())
        self.belief_sum = {s: 0.0 for s in STATES}
        self.n_steps = 0
        self._clear_window(None)

    def reset(self):
        """New exercise: start from a mix of the prior and what was learnt this session."""
        w = config.HMM_SESSION_MEMORY if self.n_steps else 0.0
        self.belief = {
            s: (1 - w) * config.HMM_PRIOR[s] + w * self.belief_sum[s] / max(self.n_steps, 1)
            for s in STATES
        }
        self._clear_window(None)

    def observe(self, t, balance, events):
        """Collect the observations of one frame. Updates the belief when a step is full.

        :param t: time in seconds
        :param balance: balance level of this frame ('low'/'moderate'/'high' or None)
        :param events: event symbols of this frame (LOW_ADHERENCE, TOO_FAST, GOOD)
        :return: True if the belief was updated in this frame
        """
        if self.window_start is None:
            self.window_start = t
        if balance is not None:
            self.window_balance[balance] += 1
        self.window_events += events
        if t - self.window_start < config.HMM_STEP_S:
            return False

        level = None
        if self.window_balance:
            level = self.window_balance.most_common(1)[0][0]
        self.update(level, self.window_events)
        self._clear_window(t)
        return True

    def update(self, balance, events):
        """One forward step of the HMM.

        :param balance: most frequent balance level in the step, or None (no evidence)
        :param events: list of event symbols seen in the step
        """
        # Predict: the user may have changed state since the last step
        predicted = {
            dest: sum(self.belief[src] * self.A[src][dest] for src in STATES)
            for dest in STATES
        }
        # Correct: weigh each state by how well it explains what was observed
        posterior = {}
        for s in STATES:
            likelihood = 1.0
            if balance is not None:
                likelihood *= config.HMM_BALANCE_EMISSION[s][balance]
            for event in events:
                likelihood *= config.HMM_EVENT_LIKELIHOOD[s][event]
            posterior[s] = predicted[s] * likelihood
        total = sum(posterior.values())
        if total > 0:
            self.belief = {s: p / total for s, p in posterior.items()}
        else:
            self.belief = predicted

        for s in STATES:
            self.belief_sum[s] += self.belief[s]
        self.n_steps += 1

    def most_likely(self):
        return max(self.belief, key=self.belief.get)

    def _clear_window(self, t):
        self.window_start = t
        self.window_balance = Counter()
        self.window_events = []


class BeliefPolicy:
    """Chooses the robot action from the HMM belief. Drop-in replacement for RulePolicy."""

    def __init__(self):
        self.hmm = UserStateHMM()

    @property
    def belief(self):
        return self.hmm.belief

    def reset(self):
        self.hmm.reset()

    def decide(self, obs):
        """:param obs: Policy.Observation
        :return: Action"""
        events = []
        if obs.adherence == 'low':
            events.append(LOW_ADHERENCE)
        if obs.execution == 'poor':
            events.append(TOO_FAST)
        elif obs.execution == 'good':
            events.append(GOOD)
        # Without tracking there is no balance evidence (balance_now is None then)
        self.hmm.observe(obs.t, obs.balance_now if obs.tracking_ok else None, events)
        b = self.hmm.belief

        # 1. Without tracking no other rule has valid data.
        if not obs.tracking_ok:
            return Action.TRACKING_LOST

        # 2-3. Safety: the user is probably unstable.
        if b['unstable'] >= config.BELIEF_SIMPLIFY_P:
            return Action.REST if obs.simplified else Action.SIMPLIFY
        if b['unstable'] >= config.BELIEF_REASSURE_P:
            if obs.exercise == 'single_leg_balance':
                return Action.REASSURE_HOLD
            return Action.REASSURE_SLOW

        # 4. Adherence: the user is probably not following the program. Unlike the rule
        #    policy this can also fire between events (e.g. a long hesitation).
        if b['unmotivated'] >= config.BELIEF_MOTIVATE_P:
            return Action.MOTIVATE

        # 5. Feedback on what was just completed.
        if obs.execution == 'poor':
            return Action.REASSURE_SLOW
        if obs.execution == 'good':
            return Action.PRAISE

        return Action.NONE
