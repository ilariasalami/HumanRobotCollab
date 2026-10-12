import time

import cv2

from coach import config
from coach import Sense
from coach import Think
from coach import Act
from coach import TestLog
from coach.Policy import Action


# Main Program Loop
def main():
    """
    Main function of the osteoporosis coach: the sense-think-act loop.

    Step 3 (current): Sense -> Think (exercise flow + rule policy) -> Act (text only).
    Robot display, speech and the study Logger come in the next steps.
    """

    sense = Sense.Sense()
    think = Think.Think()
    act = Act.Act()
    # Temporary: one CSV row per frame in data/, for tuning thresholds
    test_log = TestLog.TestLog()

    # Search and print available camera devices (may take a while to complete)
    #searchValidCameraIndexes()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    start_t = None
    previous = ('calibrate', None)   # flow state and exercise of the previous frame, for the log

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        t = time.monotonic()
        if start_t is None:
            start_t = t

        # Sense: landmarks -> angles, heights, sway (and calibration at the start)
        features = sense.process(frame, t)

        # Think: exercise flow, balance level, and the robot action chosen by the policy
        output = think.step(features, t)

        # Act: show the camera view, the robot message and the measured values
        message = act.update(frame, features, output)

        # Log state changes, tracker events and robot actions (terminal + test log)
        # Events belong to the state they happened in, i.e. the one before a state change
        clock = f'{t - start_t:7.1f}s'
        for event in output.events:
            print(f'{clock}  {previous[0]:9} {previous[1] or "-":18} event   {event}')
        if output.action is not Action.NONE:
            print(f'{clock}  {output.state:9} {output.exercise or "-":18} ROBOT   {output.action.name}: "{message}"')
        previous = (output.state, output.exercise)
        test_log.write(t, features, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('n')):
            test_log.note(f'key:{chr(key)}')
        if key == ord('q'):
            break
        elif key == ord('c'):
            sense.start_calibration()
            think.recalibrate(t)
        elif key == ord('n'):
            think.skip_exercise(t)
        elif key == ord('d'):
            act.toggle_debug()
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
