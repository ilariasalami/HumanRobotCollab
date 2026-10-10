from coach import config


# Think Component: Decision Making
# Step 2: only the balance classification. The exercise-flow state machine and the
# feedback rules are added in step 3.

LOW, MODERATE, HIGH = 'low', 'moderate', 'high'


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
