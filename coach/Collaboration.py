"""The robot also needs the user: its own wobble during the exercise.

The robot exercises along with the user. Once or twice per session it loses its balance
a little and says so. If the user encourages it (a verbal cue, see Features.user_encourages),
it thanks the user; if nobody helps, it steadies itself after a while. Support then goes
both ways: the session becomes a shared effort rather than the robot only watching the user.

Safety comes first: the robot only wobbles while the user is exercising, is not
unstable themselves and has had no safety action (reassure, simplify, rest) recently.

The wobble moments are drawn from a random generator, so the same seed gives the same
session. In the live app no RobotWobble is given to Think, so the robot never wobbles.
All numbers are in config.py (ROBOT_WOBBLE_*).
"""

from coach import config
from coach.Policy import Action


class RobotWobble:
    """Decides when the robot wobbles and how it reacts to the user's encouragement."""

    def __init__(self, rng):
        """:param rng: random.Random used to plan the wobble moments"""
        count = rng.randint(*config.ROBOT_WOBBLES)
        # Moments in seconds of exercise time (time spent in 'doing') since the start
        self.planned = sorted(rng.uniform(*config.ROBOT_WOBBLE_WINDOW_S) for _ in range(count))
        self.exercise_s = 0.0
        self.last_t = None
        self.wobbling = False
        self.since = None           # when the current wobble started
        self.helped = False         # the user encouraged during the current wobble
        self.wobbles = 0
        self.times_helped = 0

    def step(self, t, exercising, safe, encouraged, free):
        """One time step.

        :param t: time in seconds
        :param exercising: the user is doing an exercise (flow state 'doing')
        :param safe: nothing in the user's situation needs the robot's attention
        :param encouraged: the user started encouraging the robot in this frame
        :param free: no other robot message is given in this frame
        :return: Action (ROBOT_WOBBLE, THANK, ROBOT_STEADY or NONE)
        """
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if exercising:
            self.exercise_s += dt

        if self.wobbling:
            if encouraged:
                self.helped = True
            if not free:
                return Action.NONE      # react in a later frame
            if self.helped:
                self.wobbling = False
                self.times_helped += 1
                return Action.THANK
            if t - self.since >= config.ROBOT_WOBBLE_MAX_S:
                self.wobbling = False
                return Action.ROBOT_STEADY
            return Action.NONE

        if (self.planned and self.exercise_s >= self.planned[0]
                and exercising and safe and free):
            self.planned.pop(0)
            # Keep some exercise time between two wobbles
            if self.planned:
                self.planned[0] = max(self.planned[0],
                                      self.exercise_s + config.ROBOT_WOBBLE_MIN_GAP_S)
            self.wobbling = True
            self.since = t
            self.helped = False
            self.wobbles += 1
            return Action.ROBOT_WOBBLE
        return Action.NONE
