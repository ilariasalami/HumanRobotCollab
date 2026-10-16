"""What Sense measures in one frame.

Kept apart from Sense.py (which needs MediaPipe and a camera) so that the
simulation (Simulation.py) can produce the same Features without them.
"""

from dataclasses import dataclass, field


@dataclass
class Features:
    """Everything Sense measured in one frame. Values are None when not available
    (body not fully visible, or the value needs a calibration that has not happened yet)."""
    t: float
    valid: bool = False             # all required landmarks visible
    missing: list = field(default_factory=list)
    landmarks: object = None        # raw MediaPipe landmarks, only for drawing

    # Joint angles in degrees (image plane). From a frontal camera knee/hip flexion is
    # unreliable: these are logged but no rule depends on them.
    knee_l: float = None
    knee_r: float = None
    hip_l: float = None
    hip_r: float = None
    # Sideways trunk lean in degrees from vertical, positive = leaning to the person's left.
    trunk_lean: float = None

    # Calibration status
    calibrated: bool = False
    calib_progress: float = 0.0     # 0..1 while calibrating

    # Relative to calibration (None until calibrated)
    torso_ratio: float = None       # apparent torso length / baseline; < 1 hints at forward lean
    hip_drop: float = None          # hip lowering in leg lengths: ~0 standing, larger when sitting
    foot_lift: float = None         # ankle height difference in leg lengths, + = left foot higher
    lifted_foot: str = None         # 'left', 'right' or None

    # Balance (None until calibrated)
    sway: float = None              # std of hip offset over SWAY_WINDOW_S, shoulder widths
    sway_excess: float = None       # sway above the calibrated baseline (>= 0)
    sway_baseline: float = None     # calibrated sway, for display/logging
    lean_excess: float = None       # |trunk lean - calibrated lean| in degrees

    # Verbal cue: the user encourages the robot. Needs speech recognition, which the live
    # app does not have yet; the simulation (Simulation.py) sets it.
    user_encourages: bool = False
