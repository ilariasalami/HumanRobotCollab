import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from coach import config
# Features lives in its own module so the simulation can use it without
# MediaPipe; it is imported here so Sense.Features keeps working.
from coach.Features import Features  # noqa: F401


# Landmarks that must be visible for a frame to be used (full body except arms/face).
REQUIRED_LANDMARKS = {
    'left_shoulder': vision.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': vision.PoseLandmark.RIGHT_SHOULDER,
    'left_hip': vision.PoseLandmark.LEFT_HIP,
    'right_hip': vision.PoseLandmark.RIGHT_HIP,
    'left_knee': vision.PoseLandmark.LEFT_KNEE,
    'right_knee': vision.PoseLandmark.RIGHT_KNEE,
    'left_ankle': vision.PoseLandmark.LEFT_ANKLE,
    'right_ankle': vision.PoseLandmark.RIGHT_ANKLE,
}


@dataclass
class Baseline:
    """Personal reference values measured while the user stands still (calibration).
    Pixel values assume the camera and the user's distance to it do not change."""
    hip_y: float           # hip midpoint height in pixels when standing
    leg_length: float      # vertical hip-to-ankle distance in pixels
    shoulder_width: float  # pixels
    torso_length: float    # hip midpoint to shoulder midpoint, pixels
    ankle_diff: float      # natural left/right ankle height difference (leg lengths)
    sway: float            # sway while standing still (shoulder widths)
    knee_l: float
    knee_r: float
    hip_l: float
    hip_r: float
    trunk_lean: float


def calculate_angle(joint1, joint2, joint3):
    """Angle in degrees at joint2 between the segments joint2-joint1 and joint2-joint3."""
    v1 = np.asarray(joint1) - np.asarray(joint2)
    v2 = np.asarray(joint3) - np.asarray(joint2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)  # clip: rounding can push it past +-1
    return math.degrees(math.acos(cos_angle))


class Calibrator:
    """Collects samples while the user stands still and turns them into a Baseline.
    Restarts whenever the body is not fully visible or the user moves too much."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = None   # when the body became visible (settling starts)
        self.samples = []     # samples collected after settling

    def progress(self, t):
        if self.start_t is None:
            return 0.0
        total = config.CALIBRATION_SETTLE_S + config.CALIBRATION_DURATION_S
        return min(1.0, (t - self.start_t) / total)

    def update(self, sample, t):
        """Add one sample (dict, see Sense._calibration_sample) or None for an unusable frame.
        Returns a Baseline once enough still time has been collected, otherwise None."""
        if sample is None:
            self.reset()
            return None

        if self.start_t is None:
            self.start_t = t
        if t - self.start_t < config.CALIBRATION_SETTLE_S:
            return None

        if self.samples:
            first = self.samples[0]
            moved = math.hypot(sample['hip_x'] - first['hip_x'], sample['hip_y'] - first['hip_y'])
            if moved > config.CALIBRATION_STILL_TOLERANCE * first['shoulder_width']:
                # Moved too much: settle again from here
                self.reset()
                self.start_t = t
                return None
        self.samples.append(sample)

        if t - self.start_t < config.CALIBRATION_SETTLE_S + config.CALIBRATION_DURATION_S:
            return None

        def median(key):
            return float(np.median([s[key] for s in self.samples]))

        # Baseline sway: median of the same rolling-window sway that is measured live,
        # so baseline and live values are directly comparable.
        sways = [s['sway'] for s in self.samples if s['sway'] is not None]
        baseline_sway = float(np.median(sways)) if sways else 0.0
        baseline = Baseline(
            hip_y=median('hip_y'),
            leg_length=median('leg_length'),
            shoulder_width=median('shoulder_width'),
            torso_length=median('torso_length'),
            ankle_diff=median('ankle_diff'),
            sway=min(baseline_sway, config.MAX_BASELINE_SWAY),
            knee_l=median('knee_l'),
            knee_r=median('knee_r'),
            hip_l=median('hip_l'),
            hip_r=median('hip_r'),
            trunk_lean=median('trunk_lean'),
        )
        self.reset()
        return baseline


# Sense Component: detect the body with the webcam and turn it into features
class Sense:

    def __init__(self, model_path=None):
        default_model = Path(__file__).parents[1] / 'models' / 'pose_landmarker_lite.task'
        model_path = Path(model_path or default_model)
        if not model_path.is_file():
            raise FileNotFoundError(
                f'MediaPipe pose model not found at {model_path}. '
                'Restore models/pose_landmarker_lite.task before running the application.'
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.mp_pose = vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

        self.smoothed_points = {}      # landmark name -> smoothed pixel position
        self.sway_window = deque()     # (t, offset in shoulder widths)
        self.sway_reference = None     # which foot the sway is measured against

        self.calibrator = Calibrator()
        self.baseline = None

    def start_calibration(self):
        """Forget the current baseline and calibrate again."""
        self.baseline = None
        self.calibrator.reset()

    def detect_joints(self, frame, t):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # VIDEO mode needs strictly increasing timestamps in milliseconds
        timestamp_ms = max(int(t * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        return self.mp_pose.detect_for_video(image, timestamp_ms)

    def process(self, frame, t):
        """Run pose detection on one frame and compute all features.

        :param frame: BGR webcam frame
        :param t: time in seconds (time.monotonic())
        :return: Features
        """
        result = self.detect_joints(frame, t)
        features = Features(t=t, calibrated=self.baseline is not None)

        if not result.pose_landmarks:
            features.missing = ['person']
            self._update_calibration(None, features)
            return features

        landmarks = result.pose_landmarks[0]
        features.landmarks = landmarks
        features.missing = [name for name, index in REQUIRED_LANDMARKS.items()
                            if landmarks[index].visibility < config.MIN_LANDMARK_VISIBILITY]
        if features.missing:
            self._update_calibration(None, features)
            return features
        features.valid = True

        p = self._smoothed_pixel_points(landmarks, frame.shape)
        shoulder_mid = (p['left_shoulder'] + p['right_shoulder']) / 2
        hip_mid = (p['left_hip'] + p['right_hip']) / 2
        ankle_mid = (p['left_ankle'] + p['right_ankle']) / 2

        # Angles
        features.knee_l = calculate_angle(p['left_hip'], p['left_knee'], p['left_ankle'])
        features.knee_r = calculate_angle(p['right_hip'], p['right_knee'], p['right_ankle'])
        features.hip_l = calculate_angle(p['left_shoulder'], p['left_hip'], p['left_knee'])
        features.hip_r = calculate_angle(p['right_shoulder'], p['right_hip'], p['right_knee'])
        torso = shoulder_mid - hip_mid
        # Image y points down; the person's left is on the right of the (unmirrored) image.
        features.trunk_lean = math.degrees(math.atan2(torso[0], -torso[1]))

        # Body sizes in pixels, used to normalise everything else
        shoulder_width = float(np.linalg.norm(p['left_shoulder'] - p['right_shoulder']))
        leg_length = float(ankle_mid[1] - hip_mid[1])
        torso_length = float(np.linalg.norm(torso))
        # + = left ankle higher than right (image y points down)
        ankle_diff = float(p['right_ankle'][1] - p['left_ankle'][1]) / max(leg_length, 1.0)

        if self.baseline is not None:
            b = self.baseline
            features.torso_ratio = torso_length / b.torso_length
            features.hip_drop = float(hip_mid[1] - b.hip_y) / b.leg_length
            features.lean_excess = abs(features.trunk_lean - b.trunk_lean)
            features.sway_baseline = b.sway
            ankle_diff = float(p['right_ankle'][1] - p['left_ankle'][1]) / b.leg_length
            features.foot_lift = ankle_diff - b.ankle_diff
            if features.foot_lift > config.FOOT_LIFT_THRESHOLD:
                features.lifted_foot = 'left'
            elif features.foot_lift < -config.FOOT_LIFT_THRESHOLD:
                features.lifted_foot = 'right'

        # Sway: horizontal offset of the hip midpoint from the base of support
        # (both ankles, or only the standing ankle when one foot is lifted).
        if features.lifted_foot == 'left':
            support_x = p['right_ankle'][0]
        elif features.lifted_foot == 'right':
            support_x = p['left_ankle'][0]
        else:
            support_x = ankle_mid[0]
        offset_px = float(hip_mid[0] - support_x)
        # After calibration normalise by the calibrated shoulder width: the current one
        # shrinks when the user turns, which would inflate the sway.
        norm_width = self.baseline.shoulder_width if self.baseline else shoulder_width
        features.sway = self._update_sway(offset_px / max(norm_width, 1.0), features.lifted_foot,
                                          t)
        if self.baseline is not None and features.sway is not None:
            features.sway_excess = max(0.0, features.sway - self.baseline.sway)

        self._update_calibration({
            'hip_x': float(hip_mid[0]),
            'hip_y': float(hip_mid[1]),
            'leg_length': leg_length,
            'shoulder_width': shoulder_width,
            'torso_length': torso_length,
            'ankle_diff': ankle_diff,
            'sway': features.sway,
            'knee_l': features.knee_l,
            'knee_r': features.knee_r,
            'hip_l': features.hip_l,
            'hip_r': features.hip_r,
            'trunk_lean': features.trunk_lean,
        }, features)
        return features

    def _update_calibration(self, sample, features):
        """Feed the calibrator while no baseline exists yet."""
        if self.baseline is not None:
            features.calib_progress = 1.0
            return
        self.baseline = self.calibrator.update(sample, features.t)
        features.calib_progress = self.calibrator.progress(features.t)
        if self.baseline is not None:
            features.calibrated = True
            features.calib_progress = 1.0

    def _smoothed_pixel_points(self, landmarks, frame_shape):
        """Convert the required landmarks to pixel coordinates (so x and y have the same
        scale) and smooth each one with an exponential moving average."""
        h, w = frame_shape[:2]
        alpha = config.SMOOTHING_ALPHA
        for name, index in REQUIRED_LANDMARKS.items():
            point = np.array([landmarks[index].x * w, landmarks[index].y * h])
            previous = self.smoothed_points.get(name)
            self.smoothed_points[name] = (point if previous is None
                                          else alpha * point + (1 - alpha) * previous)
        return self.smoothed_points

    def _update_sway(self, offset, reference, t):
        """Add an offset sample and return its standard deviation over the last SWAY_WINDOW_S.
        The window is cleared when the reference foot changes, so lifting or putting down
        a foot does not show up as a burst of sway."""
        if reference != self.sway_reference:
            self.sway_window.clear()
            self.sway_reference = reference
        self.sway_window.append((t, offset))
        while self.sway_window and t - self.sway_window[0][0] > config.SWAY_WINDOW_S:
            self.sway_window.popleft()
        if len(self.sway_window) < config.SWAY_MIN_SAMPLES:
            return None
        return float(np.std([o for _, o in self.sway_window]))
