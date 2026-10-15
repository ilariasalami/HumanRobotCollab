"""Decision layer: turns what the robot knows about the user into one robot action.

The policy only sees a small discrete Observation and returns an Action. It does not
know about the camera, the exercise trackers' internals or the screen. This is the part
that a Bayesian network / HMM based policy can replace later: write a class with the
same reset() and decide(observation) methods and pass it to Think.
"""

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    NONE = 'none'
    INSTRUCT = 'instruct'               # flow messages: what to do next
    PRAISE = 'praise'                   # a rep/hold was done correctly
    MOTIVATE = 'motivate'               # adherence is low
    REASSURE_SLOW = 'reassure_slow'     # sit-to-stand: go slower
    REASSURE_HOLD = 'reassure_hold'     # single-leg balance: hold the chair
    SIMPLIFY = 'simplify'               # switch to the simplified version of the exercise
    REST = 'rest'                       # already simplified and still unstable: stop and rest
    TRACKING_LOST = 'tracking_lost'     # the camera cannot see the whole body


# Urgency of the actions, highest first: safety > adherence > praise.
# Think uses it to decide whether a new message may interrupt the previous one.
PRIORITY = {
    Action.TRACKING_LOST: 6,
    Action.REST: 5,
    Action.SIMPLIFY: 5,
    Action.REASSURE_SLOW: 4,
    Action.REASSURE_HOLD: 4,
    Action.MOTIVATE: 3,
    Action.PRAISE: 2,
    Action.INSTRUCT: 1,
    Action.NONE: 0,
}


@dataclass
class Observation:
    """What the policy is allowed to know in one time step (all discrete, except the time)."""
    t: float                # time in seconds (frame rate varies, so a belief model needs it)
    tracking_ok: bool       # False when the body has been out of view for a while
    exercise: str           # 'sit_to_stand' or 'single_leg_balance'
    simplified: bool        # the simplified version is being done
    balance: str            # 'low' / 'moderate' / 'high' instability that has lasted (see Think)
    balance_now: str        # balance level of this frame only (can flicker, None when unknown)
    adherence: str          # 'low' or 'ok'
    execution: str          # 'good', 'poor' (rep done but too fast) or None (nothing completed)


class RulePolicy:
    """Fixed rules in order of priority: tracking > balance (safety) > adherence > praise."""

    def reset(self):
        """Called by Think when an exercise (or its simplified version) starts.
        The rules keep no memory, so there is nothing to reset."""

    def decide(self, obs):
        """:param obs: Observation
        :return: Action"""
        # 1. Without tracking no other rule has valid data.
        if not obs.tracking_ok:
            return Action.TRACKING_LOST

        # 2. Clearly unstable: make the exercise easier. If it already is the easier
        #    version, stop and rest instead.
        if obs.balance == 'high':
            return Action.REST if obs.simplified else Action.SIMPLIFY

        # 3. A bit unstable: reassure and slow down / hold on.
        if obs.balance == 'moderate':
            if obs.exercise == 'single_leg_balance':
                return Action.REASSURE_HOLD
            return Action.REASSURE_SLOW

        # 4. The program is not followed (partial rep, too slow, inactive, hold broken).
        if obs.adherence == 'low':
            return Action.MOTIVATE

        # 5. Something was completed: too fast is less controlled, otherwise praise.
        if obs.execution == 'poor':
            return Action.REASSURE_SLOW
        if obs.execution == 'good':
            return Action.PRAISE

        return Action.NONE
