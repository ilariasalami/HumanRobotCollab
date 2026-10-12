# Act Component: Provide feedback to the user

from collections import deque

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from coach import config
from coach.Policy import Action


WHITE = (255, 255, 255)
GREY = (160, 160, 160)
GREEN = (80, 200, 80)
ORANGE = (0, 165, 255)
RED = (60, 60, 230)
DARK = (30, 30, 30)
LEVEL_COLORS = {'low': GREEN, 'moderate': ORANGE, 'high': RED}
ACTION_COLORS = {
    Action.PRAISE: GREEN,
    Action.REASSURE_SLOW: ORANGE,
    Action.REASSURE_HOLD: ORANGE,
    Action.SIMPLIFY: RED,
    Action.REST: RED,
    Action.TRACKING_LOST: RED,
}

# How long events stay visible in the debug panel (seconds)
EVENT_DISPLAY_S = 4.0


def fmt(value, digits=1):
    """Format a number for display, '-' when it is not available."""
    return '-' if value is None else f'{value:.{digits}f}'


def phrase_for(action, cue, exercise, simplified):
    """Look up what the robot says for an action in config.PHRASES.

    Tries the most specific key first: cue + exercise (+ simplified), then the cue,
    then the exercise, then the general phrase of the action.
    """
    contexts = []
    if cue and exercise:
        if simplified:
            contexts.append(f'{cue}:{exercise}:simplified')
        contexts.append(f'{cue}:{exercise}')
    if cue:
        contexts.append(cue)
    if exercise:
        contexts.append(exercise)
    contexts.append(None)
    for context in contexts:
        phrase = config.PHRASES.get((action.name, context))
        if phrase:
            return phrase
    return action.name


# Act Component: everything the user sees and hears. Contains no decision logic.
class Act:

    def __init__(self):
        self.show_debug = True
        self.recent_events = deque()    # (t, text), only for the debug panel
        self.message = None             # (text, color, shown until t): current robot message
        self.last_action = Action.NONE

    def toggle_debug(self):
        self.show_debug = not self.show_debug

    def update(self, frame, features, output):
        """Draw the camera view (with skeleton), the robot message and the debug panel.

        :param features: Features from Sense
        :param output: ThinkOutput from Think
        :return: the text of the robot message that started in this frame, or None
        """
        t = features.t
        for event in output.events:
            self.recent_events.append((t, event))
        while self.recent_events and t - self.recent_events[0][0] > EVENT_DISPLAY_S:
            self.recent_events.popleft()

        # A new action replaces the message on screen (Think already decided that it may)
        new_text = None
        if output.action is not Action.NONE:
            new_text = phrase_for(output.action, output.cue, output.exercise, output.simplified)
            color = ACTION_COLORS.get(output.action, WHITE)
            self.message = (new_text, color, t + config.MESSAGE_DISPLAY_S)
            self.last_action = output.action
        if self.message and t > self.message[2]:
            self.message = None

        if features.landmarks is not None:
            drawing_utils.draw_landmarks(
                frame,
                features.landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style(),
            )
        if config.MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)  # flip after drawing so text is not mirrored

        self._draw_header(frame, features, output)
        if self.message:
            self._draw_message(frame, self.message[0], self.message[1])

        if self.show_debug:
            panel = self._debug_panel(features, output, frame.shape[0])
            frame = np.hstack([frame, panel])

        cv2.imshow(config.WINDOW_NAME, frame)
        return new_text

    def _draw_header(self, frame, features, output):
        """Top of the camera image: where we are in the session."""
        def text(line, y, scale=0.8, color=WHITE, thickness=2):
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        if output.state == 'calibrate':
            text('Calibrating', 40)
            w = frame.shape[1]
            x0, y0, x1, y1 = 20, 60, w - 20, 85
            cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 2)
            filled = int(x0 + (x1 - x0) * features.calib_progress)
            cv2.rectangle(frame, (x0, y0), (filled, y1), GREEN, -1)
            return
        if output.state == 'done':
            text('Session finished', 40)
            return

        name = output.exercise.replace('_', ' ')
        if output.simplified:
            name += ' (simplified)'
        if output.state == 'demo':
            text('Get ready: ' + name, 40)
        elif output.status is not None:
            text(name, 40)
            text(output.status.progress_text, 80, scale=0.9)
            if output.status.countdown is not None:
                text(f'{output.status.countdown:.0f}', 160, scale=2.5, color=GREEN, thickness=4)

    def _draw_message(self, frame, message, color):
        """Robot message in a dark box at the bottom of the camera image, word-wrapped."""
        h, w = frame.shape[:2]
        font, scale, thickness, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, 30
        lines, line = [], ''
        for word in message.split():
            candidate = f'{line} {word}'.strip()
            if line and cv2.getTextSize(candidate, font, scale, thickness)[0][0] > w - 40:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)

        top = h - line_h * len(lines) - 20
        cv2.rectangle(frame, (0, top), (w, h), DARK, -1)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (20, top + 30 + i * line_h), font, scale, color, thickness, cv2.LINE_AA)

    def _debug_panel(self, f, output, height):
        panel = np.zeros((height, 340, 3), dtype=np.uint8)
        status = output.status
        lines = []

        if f.valid:
            lines.append(('tracking OK', GREEN))
        else:
            lines.append(('cannot see: ' + ', '.join(f.missing), RED))
        lines.append((f'calibrated: {f.calibrated}  ({f.calib_progress:.0%})', WHITE))
        lines.append((f'knee  L {fmt(f.knee_l, 0):>4} R {fmt(f.knee_r, 0):>4}', WHITE))
        lines.append((f'hip   L {fmt(f.hip_l, 0):>4} R {fmt(f.hip_r, 0):>4}', WHITE))
        lines.append((f'torso ratio  {fmt(f.torso_ratio, 2)}', WHITE))
        lines.append((f'hip drop     {fmt(f.hip_drop, 2)}', WHITE))
        lines.append((f'foot lift    {fmt(f.foot_lift, 2)}  ({f.lifted_foot or "-"})', WHITE))
        lines.append((f'sway excess  {fmt(f.sway_excess, 3)}  (base {fmt(f.sway_baseline, 3)})', WHITE))
        lines.append((f'trunk lean   {fmt(f.trunk_lean)}  (excess {fmt(f.lean_excess)})', WHITE))
        lines.append((f'BALANCE      {output.balance or "-"}', LEVEL_COLORS.get(output.balance, WHITE)))
        lines.append((f'  lasting    {output.balance_sustained or "-"}',
                      LEVEL_COLORS.get(output.balance_sustained, WHITE)))
        lines.append(('', WHITE))

        lines.append((f'FLOW STATE   {output.state}', WHITE))
        if output.exercise:
            lines.append((f'{output.exercise}{" (simpl.)" if output.simplified else ""}', WHITE))
        if status is not None:
            lines.append((f'phase        {status.phase}', WHITE))
            lines.append((f'progress     {status.progress_text}  ({status.progress:.0%})', WHITE))
            lines.append((f'adherence    {status.adherence:.2f}', WHITE))
        lines.append((f'last action  {self.last_action.name}', ACTION_COLORS.get(self.last_action, WHITE)))
        lines.append(('events: ' + ' '.join(e for _, e in self.recent_events), ORANGE))
        lines.append(('', WHITE))

        lines.append(('c recalibrate  n next exercise (test)', GREY))
        lines.append(('d debug  m marker in test log  q quit', GREY))

        for i, (text, color) in enumerate(lines):
            cv2.putText(panel, text, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return panel
