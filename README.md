# Osteoporosis Exercise Coach

A socially assistive virtual robot that guides weight-bearing exercises for osteoporosis
rehabilitation (University of Twente, Human-Robot Collaboration course). It is built as a
sense-think-act loop:

- **Sense**: body landmarks from a webcam (MediaPipe), turned into hip drop, foot lift,
  sway and trunk lean, relative to a short calibration.
- **Think**: exercise trackers (sit-to-stand reps, single-leg balance holds), an
  exercise-flow state machine, and a policy that chooses the robot's action: a fixed rule
  policy, or an HMM that keeps a belief over the user's hidden state
  (engaged / unmotivated / unstable).
- **Act**: the robot instructs, praises, motivates, reassures, simplifies the exercise
  or tells the user to rest. Priority: safety > adherence > praise.
- **Support goes both ways**: the robot exercises along with the user and wobbles once or
  twice per session. If the user encourages it, it thanks them; if not, it steadies
  itself. It never wobbles while the user needs attention for their own safety.

The same Think and policies run in two ways: live with a webcam (`main.py`) and in
simulation with a virtual user (`simulate.py`, `evaluate.py`).

Based on the [FIT Interactive Coaching System](https://github.com/utwente-interaction-lab/FIT-Interactive-Coaching-System)
of the UT Interaction Lab (MIT license, see `LICENSE`).

## Setup (Windows PowerShell)

Requires Python 3.14.

```powershell
py -3.14 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by the execution policy:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Simulation (no camera needed)

### One session

```powershell
python simulate.py
```

Runs a full session (calibration, sit-to-stand, single-leg balance) with a simulated user
and the HMM policy. The terminal shows every time step where something happens: the
hidden user state, the robot's belief, tracker events and the robot's action with what it
says. It writes:

- `data/sim_<policy>_<user>_seed<n>.csv`: one row per time step (true user state, measured
  values, balance level, belief, flow state, events, robot action)
- `data/sim_<policy>_<user>_seed<n>.png`: timeline of the session

Options:

| Option | Meaning |
|---|---|
| `--policy hmm` / `--policy rule` | HMM belief policy (default) or the fixed rules |
| `--user mixed` | simulated user: `steady`, `mixed` (default), `shaky`, `unmotivated` |
| `--seed 4` | random seed (default 4: a session where every key robot action occurs); the same seed gives the same session |
| `--noise 1.0` | measurement noise multiplier |
| `--fps 10` | simulated camera frame rate |
| `--every-step` | print every time step |
| `--live` | replay the session in a window (see below) |
| `--save-video` | write the replay to `data/sim_<policy>_<user>_seed<n>.mp4` (takes about 2 minutes) |
| `--speed 2` | replay speed for `--live` and `--save-video` |
| `--no-plot`, `--show` | skip the figure / also open it in a window |

### Live view

```powershell
python simulate.py --live
```

Replays the session in a window. Left, **the room** (side view): the chair, the user as a
figure that sits, stands, lifts a foot and sways, and the laptop with the webcam and the
robot's face about 3 m away (it tilts when the robot wobbles). Speech bubbles show what the
robot says and when the user encourages it; a banner marks key moments (safety actions,
the robot's wobble). Right, a card with what the robot thinks the user is (the HMM
belief), what the user actually is (simulator only), the exercise with its progress, and
the robot's body. Keys: space pause, `+` / `-` speed, `q` quit; the window can be resized.

### Comparing the policies

```powershell
python evaluate.py --seeds 20
```

Runs 20 sessions per policy for the `mixed` user at three noise levels and for each
other user profile. Prints a table (detected unstable episodes, time to the first safety
action, false alarms, time spent unstable, time to motivate, number of messages, HMM
accuracy) and writes `data/evaluation.csv` and `data/evaluation.png`. Takes 2-3 minutes.

## Live with a webcam

```powershell
python main.py
```

Stand about 3 m from the camera next to a chair, fully in view, and stand still until
calibration is done. The session then runs by itself. Keys: `c` recalibrate, `n` next
exercise (testing), `d` debug panel, `m` marker in the test log, `q` quit.

## Code structure

| File | Contents |
|---|---|
| `main.py` | Live sense-think-act loop with the webcam |
| `simulate.py` | One simulated session: log + timeline figure |
| `evaluate.py` | Rule vs HMM policy over many simulated sessions |
| `coach/config.py` | All thresholds, timings, phrases, HMM and simulation parameters |
| `coach/Sense.py` | MediaPipe landmarks -> angles, hip drop, foot lift, sway, lean; calibration |
| `coach/Exercises.py` | Sit-to-stand and single-leg balance trackers (reps, holds, adherence events) |
| `coach/Think.py` | Balance level, exercise-flow state machine, call to the policy, cooldowns |
| `coach/Policy.py` | Actions, the Observation given to a policy, the rule policy |
| `coach/Belief.py` | HMM over the user's state (forward algorithm) and the belief policy |
| `coach/LiveView.py` | Live view / video of a simulated session (drawing only) |
| `coach/Collaboration.py` | The robot's own wobble and its reaction to the user's encouragement |
| `coach/Simulation.py` | Simulated user (hidden state, movement, sensor noise, encouraging the robot), session runner, metrics |
| `coach/Act.py` | Camera view, robot message bar, debug panel |
| `coach/TestLog.py` | Temporary per-frame CSV log of the live app |

Logs and figures go in `data/`, which is not committed.
