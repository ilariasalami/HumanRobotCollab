# Act Component: Provide feedback to the user

from collections import deque

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from coach import config


WHITE = (255, 255, 255)
GREY = (160, 160, 160)
GREEN = (80, 200, 80)
ORANGE = (0, 165, 255)
RED = (60, 60, 230)
LEVEL_COLORS = {'low': GREEN, 'moderate': ORANGE, 'high': RED}

# How long events stay visible in the debug panel (seconds)
EVENT_DISPLAY_S = 4.0


def fmt(value, digits=1):
    """Format a number for display, '-' when it is not available."""
    return '-' if value is None else f'{value:.{digits}f}'


# Act Component: everything the user sees and hears. Contains no decision logic.
class Act:

    def __init__(self):
        self.show_debug = True
        self.recent_events = deque()    # (t, text), only for the debug panel

    def toggle_debug(self):
        self.show_debug = not self.show_debug

    def update(self, frame, features, balance=None, status=None):
        """Draw the camera view (with skeleton) and the debug panel, and show the window.

        :param features: Features from Sense
        :param balance: balance level from Think ('low'/'moderate'/'high' or None)
        :param status: ExerciseStatus of the current exercise, or None
        """
        if status is not None:
            for event in status.events:
                self.recent_events.append((features.t, event))
        while self.recent_events and features.t - self.recent_events[0][0] > EVENT_DISPLAY_S:
            self.recent_events.popleft()

        if features.landmarks is not None:
            drawing_utils.draw_landmarks(
                frame,
                features.landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style(),
            )
        if config.MIRROR_DISPLAY:
            frame = cv2.flip(frame, 1)  # flip after drawing so text is not mirrored

        if not features.calibrated:
            self._draw_calibration(frame, features)
        elif status is not None:
            self._draw_exercise(frame, status)

        if self.show_debug:
            panel = self._debug_panel(features, balance, status, frame.shape[0])
            frame = np.hstack([frame, panel])

        cv2.imshow(config.WINDOW_NAME, frame)

    def _draw_calibration(self, frame, features):
        h, w = frame.shape[:2]
        cv2.putText(frame, 'Stand still next to your chair, facing the camera', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        # Progress bar
        x0, y0, x1, y1 = 20, 60, w - 20, 85
        cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 2)
        filled = int(x0 + (x1 - x0) * features.calib_progress)
        cv2.rectangle(frame, (x0, y0), (filled, y1), GREEN, -1)

    def _draw_exercise(self, frame, status):
        """Big progress text (and hold countdown) on the camera image."""
        name = status.exercise.replace('_', ' ')
        if status.simplified:
            name += ' (simplified)'
        cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, status.progress_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        if status.countdown is not None:
            cv2.putText(frame, f'{status.countdown:.0f}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, GREEN, 4,
                        cv2.LINE_AA)

    def _debug_panel(self, f, balance, status, height):
        panel = np.zeros((height, 340, 3), dtype=np.uint8)
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
        lines.append(('', WHITE))
        lines.append((f'sway excess  {fmt(f.sway_excess, 3)}  (base {fmt(f.sway_baseline, 3)})', WHITE))
        lines.append((f'trunk lean   {fmt(f.trunk_lean)}  (excess {fmt(f.lean_excess)})', WHITE))
        lines.append((f'BALANCE      {balance or "-"}', LEVEL_COLORS.get(balance, WHITE)))
        lines.append(('', WHITE))

        if status is not None:
            lines.append((f'{status.exercise}{" (simpl.)" if status.simplified else ""}', WHITE))
            lines.append((f'phase        {status.phase}', WHITE))
            lines.append((f'progress     {status.progress_text}  ({status.progress:.0%})', WHITE))
            lines.append((f'adherence    {status.adherence:.2f}', WHITE))
            lines.append(('events: ' + ' '.join(e for _, e in self.recent_events), ORANGE))
            lines.append(('', WHITE))

        lines.append(('1 sit-to-stand  2 single-leg', GREY))
        lines.append(('s simplified on/off  r restart', GREY))
        lines.append(('c recalibrate  d debug  q quit', GREY))
        lines.append(('m marker in test log', GREY))

        for i, (text, color) in enumerate(lines):
            cv2.putText(panel, text, (10, 24 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return panel
