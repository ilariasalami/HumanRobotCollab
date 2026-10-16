"""Live view of a simulated session (assignment 3): the room and the robot's mind.

Replays the rows of Simulation.run_session() frame by frame:
- left, THE ROOM (side view): the chair, the user as a stick figure that sits, stands,
  lifts a foot and sways, and the laptop with the camera and the robot's face, about
  3 m away. Speech bubbles show what the robot says and when the user encourages it.
  The user's hidden state is shown small and marked "simulator only".
- right, THE ROBOT'S MIND as beliefs, desires and intentions: the HMM belief, the
  robot's fixed priorities, and its current plan and action.
- bottom: a timeline of the session.

Only drawing here, no decisions: everything shown comes from the logged rows.
Drawn with Pillow (fonts, smooth shapes); OpenCV shows the window and writes the video.
"""

import math
import textwrap
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coach import config

W, H = 1280, 720
SCENE_W = 830               # left part: the room; right part: the robot's mind
TIMELINE_Y = 600

# Colours (RGB). The three user states use the same colours as the timeline figure.
BG = (250, 249, 246)
PANEL = (240, 239, 234)
INK = (31, 31, 29)
MUTED = (110, 109, 104)
LINE = (205, 203, 196)
FLOOR = (222, 219, 210)
WOOD = (176, 141, 104)
SCREEN = (34, 40, 52)
FACE = (238, 242, 247)
CURSOR = (42, 120, 214)
STATE_COLORS = {'engaged': (42, 120, 214), 'unmotivated': (27, 175, 122),
                'unstable': (235, 104, 52)}

# Room geometry in pixels (side view, the user faces right towards the laptop)
FLOOR_Y = 500
THIGH = SHIN = 80           # leg length 160 px
TORSO, NECK, HEAD_R = 100, 8, 17
ANKLE_X = 300               # where the user stands
SEAT_Y = FLOOR_Y - 95       # top of the chair seat
CHAIR_X0, CHAIR_X1 = 190, 262
TABLE_X0, TABLE_X1, TABLE_Y = 600, 790, FLOOR_Y - 140
LAPTOP = (628, 248, 762, 350)           # screen rectangle
FACE_C = (695, 300)

# Which desire each robot action serves (1 safety, 2 adherence, 3 the shared session)
DESIRES = ['Keep the user safe', 'Follow the exercise program', 'Finish the session together']
ACTION_DESIRE = {
    'TRACKING_LOST': 0, 'REASSURE_SLOW': 0, 'REASSURE_HOLD': 0, 'SIMPLIFY': 0, 'REST': 0,
    'MOTIVATE': 1, 'INSTRUCT': 1,
    'PRAISE': 2, 'ROBOT_WOBBLE': 2, 'THANK': 2, 'ROBOT_STEADY': 2,
}
ACTION_TEXT = {
    'INSTRUCT': 'instructing', 'PRAISE': 'praising', 'MOTIVATE': 'motivating',
    'REASSURE_SLOW': 'reassuring: slow down', 'REASSURE_HOLD': 'reassuring: hold the chair',
    'SIMPLIFY': 'switching to the easier version', 'REST': 'asking the user to rest',
    'TRACKING_LOST': 'asking the user to step into view', 'ROBOT_WOBBLE': 'wobbling itself',
    'THANK': 'thanking the user for the help', 'ROBOT_STEADY': 'steadying itself',
}
SAFETY = {'TRACKING_LOST', 'REASSURE_SLOW', 'REASSURE_HOLD', 'SIMPLIFY', 'REST'}
ACTION_SHOWN_S = 3.0        # how long the current action is shown under INTENTION


def font(size, bold=False):
    try:
        return ImageFont.truetype('arialbd.ttf' if bold else 'arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size)


F_SMALL, F_TEXT, F_HEAD = font(14), font(17), font(15, bold=True)
F_TITLE, F_BUBBLE, F_BOLD = font(20, bold=True), font(18), font(17, bold=True)


def lerp(a, b, s):
    return a + (b - a) * s


def knee_position(hip, ankle):
    """Two-segment leg: the knee bends forward (towards +x)."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    d = math.hypot(dx, dy)
    if d >= THIGH + SHIN - 1e-6:
        return (lerp(hip[0], ankle[0], 0.5), lerp(hip[1], ankle[1], 0.5))
    h = math.sqrt(THIGH ** 2 - (d / 2) ** 2)
    mx, my = (hip[0] + ankle[0]) / 2, (hip[1] + ankle[1]) / 2
    nx, ny = -dy / d, dx / d            # unit normal
    if nx < 0:
        nx, ny = -nx, -ny
    return (mx + nx * h, my + ny * h)


class LiveView:
    """Draws the frames of one simulated session."""

    def __init__(self, rows, title):
        """:param rows: rows from Simulation.run_session()
        :param title: text at the top of the room view"""
        self.rows = rows
        self.title = title
        self.dt = rows[1]['t'] - rows[0]['t'] if len(rows) > 1 else 0.1
        self.has_belief = rows[0]['p_engaged'] is not None
        # Seated hip drop of this user, so 'seated' always lands on the chair
        drops = sorted(r['hip_drop'] for r in rows if r['hip_drop'] is not None)
        self.seated_drop = max(0.2, drops[int(0.97 * (len(drops) - 1))]) if drops else 0.4
        self._precompute_messages()
        self.background = self._draw_background()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _precompute_messages(self):
        """For every frame: the robot message on screen and the last action."""
        self.message = []
        self.last_action = []
        message, message_t, action, action_t = '', -math.inf, '', -math.inf
        for r in self.rows:
            if r['action']:
                message, message_t = r['message'], r['t']
                action, action_t = r['action'], r['t']
            shown = message if r['t'] - message_t < config.MESSAGE_DISPLAY_S else ''
            self.message.append(shown)
            self.last_action.append(action if r['t'] - action_t < ACTION_SHOWN_S else '')

    def _draw_background(self):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        # Robot's mind panel
        d.rectangle((SCENE_W, 0, W, H), fill=PANEL)
        d.line((SCENE_W, 0, SCENE_W, H), fill=LINE, width=1)
        # Floor
        d.rectangle((0, FLOOR_Y, SCENE_W, FLOOR_Y + 52), fill=FLOOR)
        d.line((0, FLOOR_Y, SCENE_W, FLOOR_Y), fill=MUTED, width=2)
        # Chair (behind the user)
        d.rectangle((CHAIR_X0, SEAT_Y, CHAIR_X1, SEAT_Y + 9), fill=WOOD)
        d.line((CHAIR_X0 + 4, SEAT_Y + 9, CHAIR_X0 + 4, FLOOR_Y), fill=WOOD, width=6)
        d.line((CHAIR_X1 - 4, SEAT_Y + 9, CHAIR_X1 - 4, FLOOR_Y), fill=WOOD, width=6)
        d.line((CHAIR_X0 + 4, SEAT_Y, CHAIR_X0 + 4, SEAT_Y - 100), fill=WOOD, width=7)
        d.text((CHAIR_X0 - 10, FLOOR_Y + 8), 'chair', font=F_SMALL, fill=MUTED)
        # Table with the laptop (camera on top of the screen)
        d.rectangle((TABLE_X0, TABLE_Y, TABLE_X1, TABLE_Y + 8), fill=WOOD)
        for x in (TABLE_X0 + 10, TABLE_X1 - 10):
            d.line((x, TABLE_Y + 8, x, FLOOR_Y), fill=WOOD, width=6)
        x0, y0, x1, y1 = LAPTOP
        d.polygon([(x0 - 12, TABLE_Y), (x1 + 12, TABLE_Y), (x1, y1), (x0, y1)], fill=(70, 74, 82))
        d.rounded_rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 2), radius=6, fill=(70, 74, 82))
        d.rectangle(LAPTOP, fill=SCREEN)
        d.ellipse((FACE_C[0] - 3, y0 - 4, FACE_C[0] + 3, y0 + 2), fill=(120, 220, 140))
        d.text((x0 + 4, TABLE_Y + 12), 'laptop + webcam', font=F_SMALL, fill=MUTED)
        # Distance between user and camera
        y = FLOOR_Y + 32
        d.line((ANKLE_X, y, FACE_C[0], y), fill=MUTED, width=1)
        for x, s in ((ANKLE_X, 1), (FACE_C[0], -1)):
            d.polygon([(x, y), (x + 8 * s, y - 4), (x + 8 * s, y + 4)], fill=MUTED)
        d.text(((ANKLE_X + FACE_C[0]) / 2 - 14, y + 4), 'about 3 m', font=F_SMALL, fill=MUTED)
        # Timeline frame and session blocks
        d.text((16, TIMELINE_Y + 2), 'SESSION', font=F_HEAD, fill=MUTED)
        t_end = self.rows[-1]['t'] or 1.0

        def x_of(t):
            """x position of time t on the timeline"""
            return 20 + (SCENE_W - 40) * t / t_end
        self.x_of = x_of
        y0, y1 = TIMELINE_Y + 30, TIMELINE_Y + 50
        d.rectangle((20, y0, SCENE_W - 20, y1), fill=(234, 232, 226))
        start = None
        for r in self.rows + [{'flow_state': '', 't': t_end}]:
            if r['flow_state'] == 'doing' and start is None:
                start = r['t']
            elif r['flow_state'] != 'doing' and start is not None:
                d.rectangle((x_of(start), y0, x_of(r['t']), y1), fill=(208, 205, 196))
                start = None
        for r in self.rows:
            if r['action']:
                color = STATE_COLORS['unstable'] if r['action'] in SAFETY else INK
                d.line((x_of(r['t']), y0 - 6, x_of(r['t']), y0 - 1), fill=color, width=2)
            if r['user_encourages']:
                d.ellipse((x_of(r['t']) - 3, y1 + 3, x_of(r['t']) + 3, y1 + 9),
                          fill=STATE_COLORS['unmotivated'])
        d.text((20, y1 + 14), 'grey: exercising    ticks: robot actions (orange = safety)    '
               'green dots: user encourages the robot', font=F_SMALL, fill=MUTED)
        # Mind panel headers
        d.text((SCENE_W + 24, 18), "THE ROBOT'S MIND", font=F_TITLE, fill=INK)
        return img

    # ------------------------------------------------------------------
    # One frame
    # ------------------------------------------------------------------
    def render(self, i, footer=''):
        """:return: PIL image of frame i"""
        r = self.rows[i]
        img = self.background.copy()
        d = ImageDraw.Draw(img)
        d.text((16, 14), self.title, font=F_TITLE, fill=INK)
        head = self._draw_user(d, r)
        self._draw_robot(d, r, i)
        if r['user_encourages']:
            self._bubble(d, 'You can do it, robot!', (20, head[1] - 120), 250,
                         (head[0] - 10, head[1] - HEAD_R - 4))
        if self.message[i]:
            self._bubble(d, self.message[i], (360, 90), 440, (FACE_C[0] - 10, LAPTOP[1] - 8))
        self._draw_hidden_state(d, r)
        self._draw_mind(d, r, i)
        # Timeline cursor and clock
        x = self.x_of(r['t'])
        d.line((x, TIMELINE_Y + 24, x, TIMELINE_Y + 56), fill=CURSOR, width=3)
        d.text((SCENE_W - 330, TIMELINE_Y + 2), f't = {r["t"]:5.1f} s   {footer}',
               font=F_TEXT, fill=INK)
        return img

    def _draw_user(self, d, r):
        """Stick figure from the measured hip drop, foot lift, sway and lean.
        :return: head centre"""
        t = r['t']
        drop = r['hip_drop'] if r['hip_drop'] is not None else 0.0
        s = min(max(drop / self.seated_drop, 0.0), 1.0)     # 0 standing .. 1 seated
        hip = (lerp(ANKLE_X - 6, (CHAIR_X0 + CHAIR_X1) / 2 + 8, s),
               lerp(FLOOR_Y - THIGH - SHIN + 4, SEAT_Y - 8, s))
        ankle = (ANKLE_X, FLOOR_Y)
        # Lifted foot, drawn a bit higher than measured so it is visible
        lift = abs(r['foot_lift'] or 0.0)
        lifted = lift > config.FOOT_LIFT_THRESHOLD
        lift_px = min(lift, 0.3) * (THIGH + SHIN) * 2.0 if lifted else 0.0
        ankle2 = (ANKLE_X - 0.7 * lift_px, FLOOR_Y - lift_px)

        # Torso: leans forward when sitting, wobbles with sway and lean
        lean = max(0.0, (r['lean_excess'] or 0.0) - 3.0)
        sway = r['sway_excess'] or 0.0
        angle = (18 * s + min(lean, 30) * 0.8 * math.sin(2 * math.pi * 0.8 * t)
                 + min(sway, 0.12) * 180 * math.sin(2 * math.pi * 1.4 * t))
        a = math.radians(angle)
        shoulder = (hip[0] + TORSO * math.sin(a), hip[1] - TORSO * math.cos(a))
        head = (shoulder[0] + (NECK + HEAD_R) * math.sin(a),
                shoulder[1] - (NECK + HEAD_R) * math.cos(a))

        if not r['valid']:
            body, width = (185, 183, 176), 6      # not detected by the camera this frame
        else:
            body, width = INK, 7
        # Back leg first (the lifted one), then torso, front leg, arm, head
        d.line([hip, knee_position(hip, ankle2), ankle2], fill=(120, 119, 114), width=width,
               joint='curve')
        d.line([hip, shoulder], fill=body, width=width + 1)
        d.line([hip, knee_position(hip, ankle), ankle], fill=body, width=width, joint='curve')
        d.line((ankle[0], ankle[1], ankle[0] + 16, ankle[1]), fill=body, width=width)
        if r['exercise'] == 'single_leg_balance' and r['flow_state'] == 'doing':
            hand = (CHAIR_X0 + 10, SEAT_Y - 100)       # holding the chair back
            elbow = ((shoulder[0] + hand[0]) / 2, (shoulder[1] + hand[1]) / 2 + 18)
        else:
            elbow = (shoulder[0] + 22, shoulder[1] + 38)  # arms crossed on the chest
            hand = (shoulder[0] + 6, shoulder[1] + 22)
        d.line([shoulder, elbow, hand], fill=body, width=width - 2, joint='curve')
        d.ellipse((head[0] - HEAD_R, head[1] - HEAD_R, head[0] + HEAD_R, head[1] + HEAD_R),
                  fill=BG, outline=body, width=width - 2)
        if not r['valid']:
            d.text((head[0] + 24, head[1] - 8), 'not fully in view', font=F_SMALL, fill=MUTED)
        return head

    def _draw_robot(self, d, r, i):
        """The robot's face on the laptop screen: expression from its current action."""
        action = self.last_action[i]
        tilt = math.radians(14 * math.sin(2 * math.pi * 1.2 * r['t'])) if r['robot_wobbling'] else 0
        cx, cy = FACE_C

        def p(x, y):
            return (cx + x * math.cos(tilt) - y * math.sin(tilt),
                    cy + x * math.sin(tilt) + y * math.cos(tilt))

        for ex in (-20, 20):
            x, y = p(ex, -10)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=FACE)
        if r['robot_wobbling']:
            x, y = p(0, 18)
            d.ellipse((x - 7, y - 8, x + 7, y + 8), outline=FACE, width=3)      # "whoa"
        elif action in SAFETY:
            d.line([p(-16, 20), p(16, 20)], fill=FACE, width=3)                 # concerned
            d.line([p(-28, -26), p(-12, -21)], fill=FACE, width=3)
            d.line([p(28, -26), p(12, -21)], fill=FACE, width=3)
        else:
            d.line([p(-20, 12), p(-10, 21), p(0, 24), p(10, 21), p(20, 12)], fill=FACE,
                   width=3, joint='curve')                                       # smile
        if r['robot_wobbling']:
            # Motion marks on both sides of the screen
            d.arc((cx - 100, cy - 24, cx - 76, cy + 24), 120, 240, fill=MUTED, width=2)
            d.arc((cx + 76, cy - 24, cx + 100, cy + 24), -60, 60, fill=MUTED, width=2)

    def _bubble(self, d, text, xy, width, tail):
        """Speech bubble with its tail pointing at the speaker."""
        lines = textwrap.wrap(text, width=max(10, int(width / 9.5)))
        x, y = xy
        h = 14 + 23 * len(lines)
        # Tail base: under the bubble, as close above the speaker as possible
        base = min(max(tail[0], x + 40), x + width - 40)
        d.polygon([(base - 16, y + h - 2), (base + 16, y + h - 2), tail],
                  fill=(255, 255, 255), outline=LINE)
        d.rounded_rectangle((x, y, x + width, y + h), radius=12, fill=(255, 255, 255),
                            outline=LINE, width=2)
        d.rectangle((base - 15, y + h - 3, base + 15, y + h + 1), fill=(255, 255, 255))
        for k, line in enumerate(lines):
            d.text((x + 14, y + 8 + 23 * k), line, font=F_BUBBLE, fill=INK)

    def _draw_hidden_state(self, d, r):
        y = FLOOR_Y + 62
        d.text((16, y), 'hidden user state (simulator only, the robot cannot see this):',
               font=F_SMALL, fill=MUTED)
        color = STATE_COLORS[r['user_state']]
        d.ellipse((16, y + 22, 30, y + 36), fill=color)
        d.text((38, y + 19), f'{r["user_state"]}   fatigue {r["fatigue"]:.2f}', font=F_TEXT,
               fill=INK)

    def _draw_mind(self, d, r, i):
        x = SCENE_W + 24
        y = 62
        # BELIEFS
        d.text((x, y), 'BELIEFS  what the robot thinks the user is', font=F_HEAD, fill=MUTED)
        y += 28
        if self.has_belief:
            for state in config.USER_STATES:
                p = r[f'p_{state}']
                d.text((x, y), state, font=F_TEXT, fill=INK)
                d.rectangle((x + 120, y + 3, x + 320, y + 19), fill=(226, 224, 217))
                d.rectangle((x + 120, y + 3, x + 120 + 200 * p, y + 19),
                            fill=STATE_COLORS[state])
                d.text((x + 330, y), f'{p:.2f}', font=F_TEXT, fill=INK)
                y += 28
            d.text((x, y), '(HMM, updated every 0.25 s)', font=F_SMALL, fill=MUTED)
            y += 22
        else:
            d.text((x, y), 'Rule policy: no belief, it reacts to', font=F_TEXT, fill=INK)
            d.text((x, y + 22), 'the sustained balance level and events', font=F_TEXT, fill=INK)
            y += 50
        d.text((x, y), f'balance measured now: {r["balance"] or "-"}', font=F_TEXT, fill=INK)
        y += 22
        d.text((x, y), f'balance sustained: {r["balance_sustained"] or "-"}', font=F_TEXT,
               fill=INK)
        y += 40

        # DESIRES
        d.text((x, y), "DESIRES  the robot's goals, in priority", font=F_HEAD, fill=MUTED)
        y += 28
        active = ACTION_DESIRE.get(self.last_action[i])
        for k, desire in enumerate(DESIRES):
            if k == active:
                d.rounded_rectangle((x - 8, y - 3, W - 24, y + 22), radius=6, fill=(255, 255, 255))
                d.text((x, y), f'{k + 1}  {desire}', font=F_BOLD, fill=INK)
            else:
                d.text((x, y), f'{k + 1}  {desire}', font=F_TEXT, fill=MUTED)
            y += 28
        y += 16

        # INTENTIONS
        d.text((x, y), 'INTENTION  what it is doing now', font=F_HEAD, fill=MUTED)
        y += 28
        flow = r['flow_state']
        exercise = r['exercise'].replace('_', ' ') if r['exercise'] else ''
        if r['simplified'] and exercise:
            exercise += ' (easier version)'
        plan = {'calibrate': 'calibrating: user stands still',
                'demo': f'explaining: {exercise}',
                'doing': f'exercising together: {exercise}',
                'done': 'session finished'}.get(flow, flow)
        for line in textwrap.wrap(plan, 40):
            d.text((x, y), line, font=F_TEXT, fill=INK)
            y += 22
        if r['progress'] and flow == 'doing':
            d.text((x, y), f'progress: {r["progress"]}', font=F_TEXT, fill=INK)
            y += 22
        now = ACTION_TEXT.get(self.last_action[i], 'watching the user')
        d.text((x, y), f'action: {now}', font=F_BOLD, fill=INK)
        y += 22
        d.text((x, y), 'robot body: ' + ('wobbling' if r['robot_wobbling'] else 'steady'),
               font=F_TEXT, fill=INK)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    @staticmethod
    def to_bgr(img):
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def play(self, speed=2.0, window='Simulated session'):
        """Show the session in a window. Keys: space pause, + / - speed, q quit.

        Follows the wall clock: when drawing is slower than the replay speed, frames are
        skipped instead of slowing the replay down."""
        sim_t, paused = 0.0, False
        last = time.monotonic()
        t_end = self.rows[-1]['t']
        while sim_t <= t_end:
            i = min(int(round(sim_t / self.dt)), len(self.rows) - 1)
            footer = f'speed x{speed:g}   space: pause   + / -: speed   q: quit'
            cv2.imshow(window, self.to_bgr(self.render(i, footer)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                paused = not paused
            elif key in (ord('+'), ord('=')):
                speed = min(16.0, speed * 2)
            elif key == ord('-'):
                speed = max(0.5, speed / 2)
            now = time.monotonic()
            if not paused:
                sim_t += (now - last) * speed
            last = now
        cv2.destroyAllWindows()

    def save_video(self, path, speed=2.0):
        """Write the whole session to an .mp4 file (no window)."""
        fps = speed / self.dt
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        for i in range(len(self.rows)):
            writer.write(self.to_bgr(self.render(i, f'speed x{speed:g}')))
        writer.release()
