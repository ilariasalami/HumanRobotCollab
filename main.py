import time

import cv2

from coach import config
from coach import Sense
from coach import Think
from coach import Act
from coach import TestLog
from coach.Exercises import SitToStand, SingleLegBalance


# Main Program Loop
def main():
    """
    Main function of the osteoporosis coach.

    Step 2 (current): Sense + calibration + exercise trackers, chosen by hand with keys.
    The exercise-flow state machine, the feedback rules and the Logger come in the next steps.
    """

    sense = Sense.Sense()
    act = Act.Act()
    # Temporary: one CSV row per frame in data/, for tuning thresholds
    test_log = TestLog.TestLog()

    # Step 2 test controls: which tracker is active (chosen with keys 1/2/s/r)
    exercise_class = SitToStand
    simplified = False
    tracker = None

    # Search and print available camera devices (may take a while to complete)
    #searchValidCameraIndexes()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        t = time.monotonic()

        # Sense: landmarks -> angles, heights, sway (and calibration at the start)
        features = sense.process(frame, t)

        # Think: balance level and exercise progress (only after calibration)
        balance = Think.balance_level(features)
        status = None
        if features.calibrated:
            if tracker is None:
                tracker = exercise_class(simplified, t)
            status = tracker.update(features, t)
            for event in status.events:
                print(f'{t:8.1f}  {status.exercise:20} {event}')

        # Act: show the camera view and the measured values
        act.update(frame, features, balance, status)
        test_log.write(t, features, balance, status)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('1'), ord('2'), ord('s'), ord('r')):
            test_log.note(f'key:{chr(key)}')
        if key == ord('q'):
            break
        elif key == ord('c'):
            sense.start_calibration()
            tracker = None
        elif key == ord('d'):
            act.toggle_debug()
        elif key == ord('1'):
            exercise_class, tracker = SitToStand, None
        elif key == ord('2'):
            exercise_class, tracker = SingleLegBalance, None
        elif key == ord('s'):
            simplified, tracker = not simplified, None
        elif key == ord('r'):
            tracker = None
        elif key == ord('m'):
            test_log.marker()

    cap.release()
    cv2.destroyAllWindows()
    test_log.close()


def searchValidCameraIndexes():
    # checks the first 10 indexes. May take a while to complete

    print(f"Searching available camera index nrs")
    valid_cams = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print(f"Warning: unable to open video source: {i}")
        else:
            print(f"Found valid video source: {i}")
            valid_cams.append(i)

    print(f"Available camera index nrs: {valid_cams}")

if __name__ == "__main__":
    main()
