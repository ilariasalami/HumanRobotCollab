# CLAUDE.md

Context for Claude Code. Read this before working on the repo.

## Project

University of Twente course project (Human Robot Interaction / Human Robot Collaboration), team of 4, ~3 weeks.

We are building a **socially assistive virtual robot** that runs on a laptop and guides **weight-bearing exercises for osteoporosis rehabilitation**. The robot shows exercises, tracks the user via webcam, and gives personalised feedback on posture, balance, and adherence.

**Research question:** How does a socially assistive virtual agent, acting as a collaborative partner, influence self-reported motivation, perceived safety, and a sense of shared agency in rehabilitation exercises for osteoporosis?

**Study:** small formative study, the 4 team members are the participants. Each does the routine once with a plain exercise video (baseline) and once with the robot. Only low-risk exercises.

## Current scope: basic sense-think-act logic

**Sense:** the built-in webcam records the user's movements.

**Think:** body keypoints are tracked and used to compute joint angles. These are used to estimate:
- **adherence**: how well the exercise program is followed (reps done, movement range reached, pace)
- **balance**: how stable the user is (body sway)

**Act:**
- The virtual robot shows the exercise on screen; the user copies it.
- If adherence is lower than intended → the robot gives extra motivation.
- If balance is insufficient → the robot asks the user to go slower (safety, reassurance) and shows a simplified version of the exercise.
- If the exercise is executed correctly → the robot gives motivation/praise.

Priority when several apply: **balance (safety) > low adherence > correct execution**.

## Status

- Steps 1-2 (Sense, calibration, exercise trackers): done, tested on two webcams.
- Step 3 (exercise-flow state machine in `coach/Think.py`, rule policy in `coach/Policy.py`, robot messages as text): written and checked with simulated sessions and a replay of a recorded webcam session; still needs a live webcam test.
- `coach/TestLog.py` is a temporary per-frame CSV log for tuning (not the study Logger of step 5).
- Assignment 3 (branch `feature/simulation-hmm`): `coach/Belief.py` (HMM forward filter over engaged / unmotivated / unstable + `BeliefPolicy`), `coach/Simulation.py` (simulated user, session runner, metrics), `simulate.py` (one session: log + timeline figure), `evaluate.py` (rule vs HMM over many seeds). HMM and simulation parameters are in `coach/config.py`. Still to do: reflection document (2-4 pages, Arial 11).
- The policy interface is fixed: `reset()` + `decide(Observation) -> Action`; Observation carries the time, the sustained and the per-frame balance level. Any policy (rules, HMM) plugs into `Think(policy=...)`.

## Work split after merging step 3 into `main`

Both parts branch from `main` once step 3 is merged:
- **Assignment 3 (simulation)**: simulated user with hidden state that feeds Think, HMM belief layer + belief-based policy, `simulate.py` (headless, seeded, logs every step, timeline plot), reflection. Mostly new files.
- **Assignment 4 (live study)**: live webcam test of step 3, step 4 (robot panel + speech), step 5 (study Logger, command-line arguments), video baseline condition.

## Later (NOT now — do not implement unless asked)

- Voice commands from the user ("slow down", "skip", "easier"…) with Vosk
- Robot asking for confirmation before switching exercise

## Codebase

Based on the course's example repo (FIT Interactive Coaching System, UT Interaction Lab, MIT license — keep the LICENSE file). Original: elbow flexion/extension tracking with a balloon animation, structured as sense-think-act.

| File | Original | Our target |
|---|---|---|
| `coach/Sense.py` | MediaPipe, elbow angle | knee, hip, trunk angles + balance/sway measure, short calibration at start |
| `coach/Think.py` | `transitions` FSM for flexion/extension | exercise-flow FSM (demo → doing → next exercise / simplified version), rep counting, adherence + balance rules |
| new `coach/Policy.py` | — | `RulePolicy.decide(observation) -> Action`; an HMM belief policy (assignment 3) uses the same interface |
| `coach/Act.py` | balloon + pyttsx3 | robot on screen showing the exercise, spoken + visual feedback |
| new `coach/Logger.py` | — | CSV event log |
| `main.py` | main loop | wires everything together |

## Technical decisions

- **Python 3.14** with MediaPipe 1.x (`mediapipe.tasks` API, as in the upstream example). Virtual env in `venv/` (`py -3.14 -m venv venv`). Pin versions in `requirements.txt`.
- **TTS:** pyttsx3 in its own thread with a `queue.Queue` (`runAndWait()` blocks the camera loop otherwise). Add a cooldown so feedback isn't repeated every frame.
- **Thresholds** (angle ranges, balance limits, target reps, pace) in one config file/dict, not scattered in code — they are not clinically validated and we will tune them.
- Keep the sense / think / act separation: no decision logic in Act, no rendering in Think.

## Logging

One CSV row per event: timestamp, participant ID (P1–P4, never names), condition (video/robot), exercise, joint angles, balance value, adherence value, robot action. Logs go in `data/` (gitignored). No webcam recordings in the repo.

## Conventions

- Feature branches, merge to `main` via pull request; `main` must always run.
- Do not commit `venv/`, `data/`, large model files, `*.csv`, `*.mp4`.