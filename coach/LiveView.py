"""Live view of a simulated session (assignment 3): the room and what the robot thinks.

Replays the rows of Simulation.run_session() frame by frame:
- left, the room (side view): a living room with a chair, the user as a figure that sits,
  stands, lifts a foot and sways, and the laptop with the webcam and the robot's face
  about 3 m away. Speech bubbles show what the robot says and when the user encourages
  it; a short banner marks key moments (safety actions, the robot's wobble).
- right, a card with what the robot thinks of the user (the HMM belief), the user's
  actual hidden state (simulator only), the exercise and progress, and the robot's body.

Only drawing here, no decisions: everything shown comes from the logged rows.
Drawn with Pillow (fonts, smooth shapes); OpenCV shows the window and writes the video.
"""

import math
import re
import textwrap
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coach import config

W, H = 1280, 720
RW = 930                    # width of the room; the card is on the right

# Colours (RGB). The user states use the same colours as the timeline figure.
WALL = (244, 240, 232)
FLOOR = (214, 196, 170)
FLOOR_EDGE = (190, 170, 142)
RUG = (196, 172, 146)
PANEL = (236, 234, 228)
CARD = (255, 255, 255)
INK = (31, 31, 29)
MUTED = (112, 110, 104)
LINE = (214, 211, 203)
TRACK = (232, 229, 222)
WOOD = (166, 124, 82)
BODY = (45, 51, 64)
BODY_BACK = (120, 126, 138)
SKIN = (241, 214, 189)
SCREEN = (30, 36, 48)
BEZEL = (72, 76, 86)
FACE = (236, 241, 248)
STATE_COLORS = {'engaged': (42, 120, 214), 'unmotivated': (27, 175, 122),
                'unstable': (235, 104, 52)}
GREEN = (27, 175, 122)
ORANGE = (235, 104, 52)
BLUE = (42, 120, 214)
DARK = (61, 61, 58)

# Room geometry in pixels (side view: the user faces right, towards the laptop)
FLOOR_Y = 560
THIGH = SHIN = 95
TORSO, NECK, HEAD_R = 120, 8, 21
ANKLE_X = 380
SEAT_Y = FLOOR_Y - 110
CHAIR_X0, CHAIR_X1 = 255, 335
TABLE_X0, TABLE_X1, TABLE_Y = 640, 900, FLOOR_Y - 160
LAPTOP_CX = 770
SCREEN_BOX = (-95, -145, 95, -15)        # screen relative to the laptop hinge
HINGE = (LAPTOP_CX, TABLE_Y - 8)
FACE_OFFSET = (0, -82)                   # face centre relative to the hinge

# Short banners for key moments: action -> (text, colour)
BANNERS = {
    'SIMPLIFY': ('Safety: switching to the easier version', ORANGE),
    'REST': ('Safety: the robot asks the user to rest', ORANGE),
    'TRACKING_LOST': ('The camera cannot see the whole body', DARK),
    'ROBOT_WOBBLE': ('The robot wobbles', BLUE),
    'THANK': ('The user helped the robot', GREEN),
    'ROBOT_STEADY': ('Nobody helped: the robot steadies itself', DARK),
}
BANNER_S = 3.0
SAFETY = {'TRACKING_LOST', 'REASSURE_SLOW', 'REASSURE_HOLD', 'SIMPLIFY', 'REST'}
EXERCISE_NAMES = {'sit_to_stand': 'Sit to stand', 'single_leg_balance': 'Single-leg balance'}


def font(size, bold=False):
    for name in (('segoeuib.ttf', 'arialbd.ttf') if bold else ('segoeui.ttf', 'arial.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


F_SMALL, F_TEXT, F_LABEL = font(14), font(17), font(13, bold=True)
F_BIG, F_TITLE = font(21, bold=True), font(18, bold=True)
F_BUBBLE, F_KEY, F_BANNER = font(19), font(14, bold=True), font(17, bold=True)
F_PAUSED = font(26, bold=True)


def lerp(a, b, s):
    return a + (b - a) * s


def rotate(point, pivot, angle):
    x, y = point[0] - pivot[0], point[1] - pivot[1]
    c, s = math.cos(angle), math.sin(angle)
    return (pivot[0] + x * c - y * s, pivot[1] + x * s + y * c)


def knee_position(hip, ankle):
    """Two-segment leg: the knee bends forward (towards +x)."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    d = math.hypot(dx, dy)
    if d >= THIGH + SHIN - 1e-6:
        return ((hip[0] + ankle[0]) / 2, (hip[1] + ankle[1]) / 2)
    h = math.sqrt(THIGH ** 2 - (d / 2) ** 2)
    nx, ny = -dy / d, dx / d
    if nx < 0:
        nx, ny = -nx, -ny
    return ((hip[0] + ankle[0]) / 2 + nx * h, (hip[1] + ankle[1]) / 2 + ny * h)


def limb(d, points, color, width):
    """Thick line with round joints and ends."""
    d.line(points, fill=color, width=width, joint='curve')
    r = width / 2
    for x, y in (points[0], points[-1]):
        d.ellipse((x - r, y - r, x + r, y + r), fill=color)


def clock(t):
    return f'{int(t // 60)}:{int(t % 60):02d}'


class LiveView:
    """Draws the frames of one simulated session."""

    def __init__(self, rows, title):
        """:param rows: rows from Simulation.run_session()
        :param title: text at the top of the room"""
        self.rows = rows
        self.title = title
        self.dt = rows[1]['t'] - rows[0]['t'] if len(rows) > 1 else 0.1
        self.has_belief = rows[0]['p_engaged'] is not None
        # Seated hip drop of this user, so 'seated' always lands on the chair
        drops = sorted(r['hip_drop'] for r in rows if r['hip_drop'] is not None)
        self.seated_drop = max(0.2, drops[int(0.97 * (len(drops) - 1))]) if drops else 0.4
        self._precompute()
        self.background = self._draw_background()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _precompute(self):
        """For every frame: the robot message on screen, the last action and the banner."""
        self.message, self.last_action, self.banner = [], [], []
        message, message_t, action, action_t = '', -math.inf, '', -math.inf
        banner, banner_t = None, -math.inf
        for r in self.rows:
            if r['action']:
                message, message_t = r['message'], r['t']
                action, action_t = r['action'], r['t']
                if r['action'] in BANNERS:
                    banner, banner_t = BANNERS[r['action']], r['t']
            self.message.append(message if r['t'] - message_t < config.MESSAGE_DISPLAY_S else '')
            self.last_action.append(action if r['t'] - action_t < BANNER_S else '')
            self.banner.append(banner if r['t'] - banner_t < BANNER_S else None)

    def _draw_background(self):
        img = Image.new('RGB', (W, H), WALL)
        d = ImageDraw.Draw(img)
        # Window on the wall
        d.rounded_rectangle((70, 130, 240, 320), radius=6, fill=(255, 255, 255))
        d.rectangle((82, 142, 228, 308), fill=(206, 226, 240))
        d.line((155, 142, 155, 308), fill=(255, 255, 255), width=6)
        d.line((82, 225, 228, 225), fill=(255, 255, 255), width=6)
        # Floor, skirting board, rug
        d.rectangle((0, FLOOR_Y, RW, H), fill=FLOOR)
        d.rectangle((0, FLOOR_Y - 10, RW, FLOOR_Y), fill=(250, 248, 243))
        d.line((0, FLOOR_Y, RW, FLOOR_Y), fill=FLOOR_EDGE, width=2)
        d.ellipse((170, FLOOR_Y + 14, 520, FLOOR_Y + 54), fill=RUG)
        # Chair (behind the user)
        d.rounded_rectangle((CHAIR_X0, SEAT_Y, CHAIR_X1, SEAT_Y + 12), radius=4, fill=WOOD)
        for x in (CHAIR_X0 + 6, CHAIR_X1 - 6):
            d.line((x, SEAT_Y + 12, x, FLOOR_Y + 6), fill=WOOD, width=8)
        d.line((CHAIR_X0 + 6, SEAT_Y, CHAIR_X0 + 6, SEAT_Y - 125), fill=WOOD, width=9)
        # Table
        d.rounded_rectangle((TABLE_X0, TABLE_Y, TABLE_X1, TABLE_Y + 12), radius=4, fill=WOOD)
        for x in (TABLE_X0 + 14, TABLE_X1 - 14):
            d.line((x, TABLE_Y + 12, x, FLOOR_Y + 6), fill=WOOD, width=8)
        d.text((TABLE_X0 + 20, TABLE_Y + 22), 'laptop with webcam', font=F_SMALL, fill=MUTED)
        # Distance between the user and the camera
        y = FLOOR_Y + 100
        d.line((ANKLE_X, y, LAPTOP_CX, y), fill=MUTED, width=2)
        for x, s in ((ANKLE_X, 1), (LAPTOP_CX, -1)):
            d.polygon([(x, y), (x + 10 * s, y - 5), (x + 10 * s, y + 5)], fill=MUTED)
        d.text(((ANKLE_X + LAPTOP_CX) / 2 - 30, y + 6), 'about 3 m', font=F_SMALL, fill=MUTED)
        # Title
        d.text((24, 18), self.title, font=F_TITLE, fill=INK)
        # Card
        d.rectangle((RW, 0, W, H), fill=PANEL)
        d.rounded_rectangle((RW + 18, 18, W - 18, H - 18), radius=14, fill=CARD)
        return img

    # ------------------------------------------------------------------
    # One frame
    # ------------------------------------------------------------------
    def render(self, i, speed=2.0, paused=False, interactive=True):
        """:return: PIL image of frame i"""
        r = self.rows[i]
        img = self.background.copy()
        d = ImageDraw.Draw(img)
        head = self._draw_user(d, r)
        self._draw_laptop(d, r, i)
        if r['user_encourages']:
            self._bubble(d, 'You can do it, robot!', (30, head[1] - 140), 260,
                         (head[0], head[1] - HEAD_R - 6))
        if self.message[i]:
            self._bubble(d, self.message[i], (440, 120), 470, (LAPTOP_CX - 20, HINGE[1] - 150))
        if self.banner[i]:
            self._banner(d, *self.banner[i])
        self._draw_card(d, r, speed, interactive)
        if paused:
            img = self._paused(img)
        return img

    def _draw_user(self, d, r):
        """Figure from the measured hip drop, foot lift, sway and lean.
        :return: head centre"""
        t = r['t']
        drop = r['hip_drop'] if r['hip_drop'] is not None else 0.0
        s = min(max(drop / self.seated_drop, 0.0), 1.0)     # 0 standing .. 1 seated
        hip = (lerp(ANKLE_X - 8, (CHAIR_X0 + CHAIR_X1) / 2 + 10, s),
               lerp(FLOOR_Y - THIGH - SHIN + 6, SEAT_Y - 12, s))
        ankle = (ANKLE_X, FLOOR_Y)
        lift = abs(r['foot_lift'] or 0.0)
        lift_px = min(lift, 0.3) * (THIGH + SHIN) * 2.0 if lift > config.FOOT_LIFT_THRESHOLD else 0
        ankle2 = (ANKLE_X - 0.7 * lift_px, FLOOR_Y - lift_px)

        # Torso: leans forward when sitting, wobbles with the measured sway and lean
        lean = max(0.0, (r['lean_excess'] or 0.0) - 3.0)
        sway = r['sway_excess'] or 0.0
        angle = (18 * s + min(lean, 30) * 0.8 * math.sin(2 * math.pi * 0.8 * t)
                 + min(sway, 0.12) * 180 * math.sin(2 * math.pi * 1.4 * t))
        a = math.radians(angle)
        shoulder = (hip[0] + TORSO * math.sin(a), hip[1] - TORSO * math.cos(a))
        head = (shoulder[0] + (NECK + HEAD_R) * math.sin(a),
                shoulder[1] - (NECK + HEAD_R) * math.cos(a))

        seen = r['valid']
        body = BODY if seen else (178, 180, 186)
        back = BODY_BACK if seen else (205, 206, 210)
        if r['exercise'] == 'single_leg_balance' and r['flow_state'] == 'doing':
            hand = (CHAIR_X0 + 12, SEAT_Y - 122)        # holding the back of the chair
            elbow = ((shoulder[0] + hand[0]) / 2, (shoulder[1] + hand[1]) / 2 + 22)
        else:
            elbow = (shoulder[0] + 26, shoulder[1] + 44)  # arms crossed on the chest
            hand = (shoulder[0] + 8, shoulder[1] + 26)
        limb(d, [hip, knee_position(hip, ankle2), ankle2], back, 13)
        limb(d, [hip, shoulder], body, 20)
        limb(d, [hip, knee_position(hip, ankle), ankle], body, 13)
        limb(d, [ankle, (ankle[0] + 20, ankle[1])], body, 10)
        limb(d, [shoulder, elbow, hand], body, 10)
        d.ellipse((head[0] - HEAD_R, head[1] - HEAD_R, head[0] + HEAD_R, head[1] + HEAD_R),
                  fill=SKIN if seen else (226, 226, 228), outline=body, width=3)
        if not seen:
            d.text((head[0] + 30, head[1] - 10), 'not fully in view', font=F_SMALL, fill=MUTED)
        return head

    def _draw_laptop(self, d, r, i):
        """Laptop on the table; the robot's face on the screen. Tilts when the robot wobbles."""
        t = r['t']
        wobbling = r['robot_wobbling']
        angle = math.radians(8 * math.sin(2 * math.pi * 1.2 * t)) if wobbling else 0.0
        hx, hy = HINGE

        def p(x, y):
            return rotate((hx + x, hy + y), HINGE, angle)

        # Base on the table (does not tilt), then the screen
        d.polygon([(hx - 110, TABLE_Y), (hx + 110, TABLE_Y), (hx + 100, hy), (hx - 100, hy)],
                  fill=BEZEL)
        x0, y0, x1, y1 = SCREEN_BOX
        d.polygon([p(x0 - 8, y0 - 8), p(x1 + 8, y0 - 8), p(x1 + 8, y1 + 8), p(x0 - 8, y1 + 8)],
                  fill=BEZEL)
        d.polygon([p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)], fill=SCREEN)
        cx, cy = p(0, y0 - 4)
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(120, 220, 140))   # webcam light

        fx, fy = FACE_OFFSET
        action = self.last_action[i]
        blink = (t % 4.0) < 0.15 and not wobbling
        for ex in (-30, 30):
            x, y = p(fx + ex, fy - 14)
            if blink:
                d.line((x - 9, y, x + 9, y), fill=FACE, width=3)
            else:
                d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=FACE)
        if wobbling:
            x, y = p(fx, fy + 26)
            d.ellipse((x - 10, y - 12, x + 10, y + 12), outline=FACE, width=4)   # "whoa"
            for side in (-1, 1):
                mx, my = p(side * 128, -80)
                d.arc((mx - 14, my - 30, mx + 14, my + 30),
                      120 if side < 0 else -60, 240 if side < 0 else 60, fill=MUTED, width=3)
        elif action in SAFETY:
            d.line([p(fx - 24, fy + 30), p(fx + 24, fy + 30)], fill=FACE, width=4)  # concerned
            d.line([p(fx - 42, fy - 38), p(fx - 18, fy - 31)], fill=FACE, width=4)
            d.line([p(fx + 42, fy - 38), p(fx + 18, fy - 31)], fill=FACE, width=4)
        else:
            d.line([p(fx - 30, fy + 18), p(fx - 15, fy + 31), p(fx, fy + 35),
                    p(fx + 15, fy + 31), p(fx + 30, fy + 18)], fill=FACE, width=4,
                   joint='curve')                                                 # smile

    def _bubble(self, d, text, xy, width, tail):
        """Speech bubble with its tail pointing at the speaker."""
        lines = textwrap.wrap(text, width=max(10, int(width / 10)))
        x, y = xy
        h = 18 + 26 * len(lines)
        base = min(max(tail[0], x + 40), x + width - 40)
        d.polygon([(base - 16, y + h - 2), (base + 16, y + h - 2), tail],
                  fill=CARD, outline=LINE)
        d.rounded_rectangle((x, y, x + width, y + h), radius=14, fill=CARD, outline=LINE,
                            width=2)
        d.rectangle((base - 15, y + h - 3, base + 15, y + h + 1), fill=CARD)
        for k, line in enumerate(lines):
            d.text((x + 16, y + 9 + 26 * k), line, font=F_BUBBLE, fill=INK)

    def _banner(self, d, text, color):
        w = d.textlength(text, font=F_BANNER) + 40
        x = (RW - w) / 2
        d.rounded_rectangle((x, 62, x + w, 98), radius=18, fill=color)
        d.text((x + 20, 68), text, font=F_BANNER, fill=(255, 255, 255))

    def _draw_card(self, d, r, speed, interactive):
        x0, x1 = RW + 44, W - 44
        y = 40
        # What the robot thinks (HMM belief)
        d.text((x0, y), 'ROBOT THINKS THE USER IS', font=F_LABEL, fill=MUTED)
        y += 28
        if self.has_belief:
            for state in config.USER_STATES:
                p = r[f'p_{state}']
                d.text((x0, y), state, font=F_TEXT, fill=INK)
                value = f'{100 * p:.0f}%'
                d.text((x1 - d.textlength(value, font=F_TEXT), y), value, font=F_TEXT, fill=INK)
                d.rounded_rectangle((x0, y + 28, x1, y + 38), radius=5, fill=TRACK)
                if p > 0.01:
                    d.rounded_rectangle((x0, y + 28, x0 + (x1 - x0) * p, y + 38), radius=5,
                                        fill=STATE_COLORS[state])
                y += 54
        else:
            for line in ('Rule policy: no belief.', 'It reacts to the balance',
                         'level and events.'):
                d.text((x0, y), line, font=F_TEXT, fill=INK)
                y += 24
            y += 18
        # The hidden state
        y += 6
        d.text((x0, y), 'ACTUALLY  (simulator only)', font=F_LABEL, fill=MUTED)
        y += 26
        color = STATE_COLORS[r['user_state']]
        d.ellipse((x0, y + 5, x0 + 16, y + 21), fill=color)
        d.text((x0 + 26, y), r['user_state'], font=F_BIG, fill=INK)
        y += 50
        d.line((x0, y, x1, y), fill=LINE, width=1)
        y += 18

        # Exercise and progress
        d.text((x0, y), 'EXERCISE', font=F_LABEL, fill=MUTED)
        y += 24
        name = EXERCISE_NAMES.get(r['exercise'], '')
        status = {'calibrate': 'Getting ready', 'done': 'Session finished'}.get(r['flow_state'])
        d.text((x0, y), status or name, font=F_BIG, fill=INK)
        y += 32
        if r['simplified'] and not status:
            d.text((x0, y), 'easier version', font=F_TEXT, fill=MUTED)
            y += 26
        if r['flow_state'] == 'demo':
            d.text((x0, y), 'listening to the instruction', font=F_TEXT, fill=MUTED)
            y += 26
        elif r['flow_state'] == 'doing' and r['progress_frac'] is not None:
            y = self._progress_dots(d, r, x0, y)
        y += 10
        d.line((x0, y, x1, y), fill=LINE, width=1)
        y += 18

        # The robot's body
        d.text((x0, y), 'ROBOT BODY', font=F_LABEL, fill=MUTED)
        y += 26
        wobbling = r['robot_wobbling']
        d.ellipse((x0, y + 5, x0 + 16, y + 21), fill=ORANGE if wobbling else GREEN)
        d.text((x0 + 26, y), 'wobbling' if wobbling else 'steady', font=F_BIG, fill=INK)

        # Clock and controls at the bottom of the card
        y = H - 150
        d.line((x0, y, x1, y), fill=LINE, width=1)
        d.text((x0, y + 12), f'{clock(r["t"])}  of  {clock(self.rows[-1]["t"])}',
               font=F_TEXT, fill=INK)
        speed_text = f'speed x{speed:g}'
        d.text((x1 - d.textlength(speed_text, font=F_TEXT), y + 12), speed_text, font=F_TEXT,
               fill=MUTED)
        if interactive:
            # Two columns of key caps
            col = (x1 - x0) / 2
            for k, (key, label) in enumerate((('Space', 'pause'), ('Q', 'quit'),
                                              ('+', 'faster'), ('−', 'slower'))):
                self._key(d, x0 + col * (k % 2), y + 48 + 40 * (k // 2), key, label)

    def _progress_dots(self, d, r, x, y):
        """Filled and empty dots for reps or holds, with the text under them.
        :return: y below"""
        if r['exercise'] == 'sit_to_stand':
            total = (config.STS_TARGET_REPS_SIMPLIFIED if r['simplified']
                     else config.STS_TARGET_REPS)
            groups = [total]
        else:
            groups = [config.SLB_HOLDS_PER_LEG] * 2      # one group of holds per leg
            total = sum(groups)
        done = round(r['progress_frac'] * total)
        k = 0
        dx = x
        for size in groups:
            for _ in range(size):
                d.ellipse((dx, y, dx + 18, y + 18), fill=BLUE if k < done else TRACK)
                dx += 26
                k += 1
            dx += 14
        y += 28
        text = r['progress'] or ''
        text = text.replace('leg 1', 'first leg').replace('leg 2', 'second leg')
        text = re.sub(r'^Reps (\d+)/(\d+)$', r'\1 of \2 reps', text)
        text = re.sub(r'^(.+) leg: hold (\d+)/(\d+)$', r'\1 leg, hold \2 of \3', text)
        d.text((x, y), text[:1].upper() + text[1:], font=F_TEXT, fill=MUTED)
        return y + 28

    def _key(self, d, x, y, key, label):
        """A key cap with its meaning."""
        w = max(30, d.textlength(key, font=F_KEY) + 18)
        d.rounded_rectangle((x, y, x + w, y + 28), radius=6, fill=TRACK, outline=LINE)
        d.text((x + (w - d.textlength(key, font=F_KEY)) / 2, y + 5), key, font=F_KEY, fill=INK)
        d.text((x + w + 8, y + 4), label, font=F_SMALL, fill=MUTED)

    @staticmethod
    def _paused(img):
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle((RW / 2 - 200, H / 2 - 45, RW / 2 + 200, H / 2 + 45), radius=16,
                            fill=(20, 20, 20, 200))
        for text, f, y, color in (('PAUSED', F_PAUSED, H / 2 - 36, (255, 255, 255)),
                                  ('press Space to continue', F_TEXT, H / 2 + 6,
                                   (225, 225, 225))):
            d.text((RW / 2 - d.textlength(text, font=f) / 2, y), text, font=f, fill=color)
        return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    @staticmethod
    def to_bgr(img):
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fit_window(window):
        """Resizable window, as large as fits on the screen with the right aspect ratio."""
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            scale = min(0.9 * sw / W, 0.85 * sh / H)
            cv2.resizeWindow(window, int(W * scale), int(H * scale))
        except (AttributeError, OSError):
            cv2.resizeWindow(window, W, H)

    def play(self, speed=2.0, window='Simulated session'):
        """Show the session in a window. Keys: space pause, + / - speed, q quit.

        Follows the wall clock: when drawing is slower than the replay speed, frames are
        skipped instead of slowing the replay down."""
        self._fit_window(window)
        sim_t, paused = 0.0, False
        last = time.monotonic()
        t_end = self.rows[-1]['t']
        while sim_t <= t_end:
            i = min(int(round(sim_t / self.dt)), len(self.rows) - 1)
            cv2.imshow(window, self.to_bgr(self.render(i, speed, paused)))
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
            writer.write(self.to_bgr(self.render(i, speed, interactive=False)))
        writer.release()
