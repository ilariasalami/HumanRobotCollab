"""Temporary test log for tuning thresholds on a real camera (step 2).

Writes one CSV row per camera frame with every value Sense measured, the balance
level from Think and the state of the current exercise tracker. The file goes in
data/ (ignored by git). This is a testing aid, not the event Logger planned for the
study: remove it, or replace it with coach/Logger.py, once the thresholds are tuned.
"""

import csv
import time
from pathlib import Path

COLUMNS = [
    't',              # seconds since the program started
    'note',           # key presses and markers (e.g. 'key:1', 'marker:3')
    'events',         # tracker events in this frame, separated by '|'
    'valid', 'missing', 'calibrated',
    'exercise', 'simplified', 'phase', 'progress_text', 'progress', 'adherence', 'countdown',
    'balance',
    'sway', 'sway_excess', 'sway_baseline', 'trunk_lean', 'lean_excess',
    'hip_drop', 'foot_lift', 'lifted_foot', 'torso_ratio',
    'knee_l', 'knee_r', 'hip_l', 'hip_r',
]


def _fmt(value):
    """Empty cell for missing values, 4 decimals for floats."""
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return value


class TestLog:

    def __init__(self, folder=None):
        folder = Path(folder or Path(__file__).parents[1] / 'data')
        folder.mkdir(exist_ok=True)
        self.path = folder / f'testlog_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.start_t = None
        self.pending_notes = []
        self.marker_count = 0
        print(f'Test log: {self.path}')

    def note(self, text):
        """Remember a note (e.g. a key press); it is written with the next frame."""
        self.pending_notes.append(text)

    def marker(self):
        """Add a numbered marker, to find a moment back in the log."""
        self.marker_count += 1
        self.note(f'marker:{self.marker_count}')
        print(f'Marker {self.marker_count}')

    def write(self, t, features, balance, status):
        """Write one row for this frame.

        :param features: Features from Sense
        :param balance: balance level from Think, or None
        :param status: ExerciseStatus of the current exercise, or None
        """
        if self.start_t is None:
            self.start_t = t
        row = {
            't': round(t - self.start_t, 3),
            'note': '|'.join(self.pending_notes),
            'events': '|'.join(status.events) if status else '',
            'missing': '|'.join(features.missing),
            'balance': balance,
        }
        self.pending_notes = []
        for name in ('valid', 'calibrated', 'sway', 'sway_excess', 'sway_baseline', 'trunk_lean',
                     'lean_excess', 'hip_drop', 'foot_lift', 'lifted_foot', 'torso_ratio',
                     'knee_l', 'knee_r', 'hip_l', 'hip_r'):
            row[name] = getattr(features, name)
        if status is not None:
            for name in ('exercise', 'simplified', 'phase', 'progress_text', 'progress',
                         'adherence', 'countdown'):
                row[name] = getattr(status, name)
        self.writer.writerow([_fmt(row.get(column)) for column in COLUMNS])
        self.file.flush()   # so nothing is lost if the program is stopped with Ctrl+C

    def close(self):
        self.file.close()
