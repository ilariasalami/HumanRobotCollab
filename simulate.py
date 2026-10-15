"""Run one simulated coaching session (assignment 3).

A simulated user (coach/Simulation.py) replaces the webcam. The rest is the real coach:
the exercise trackers and the exercise-flow state machine (coach/Think.py) and a policy
that picks the robot's actions, either the fixed rules (coach/Policy.py) or the HMM
belief policy (coach/Belief.py).

Every time step is written to a CSV log; the terminal shows every step in which
something happens (hidden user state, flow state, tracker events, robot action).
At the end a timeline figure is saved next to the log.

Examples:
    python simulate.py                          # HMM policy, 'mixed' user, seed 0
    python simulate.py --policy rule --user shaky --seed 3
    python simulate.py --every-step --no-plot   # print all time steps
"""

import argparse
import csv
from pathlib import Path

from coach import config
from coach.Simulation import run_session, summarize

DATA_DIR = Path(__file__).parent / 'data'

# Colours of the hidden user states (validated categorical palette, colour-blind safe
# for three series). Every state is also named in the legend.
STATE_COLORS = {'engaged': '#2a78d6', 'unmotivated': '#1baf7a', 'unstable': '#eb6834'}
INK = '#52514e'
GRID = '#e4e3df'


def parse_args():
    parser = argparse.ArgumentParser(description='Simulated session of the osteoporosis coach.')
    parser.add_argument('--policy', choices=['hmm', 'rule'], default='hmm',
                        help='decision layer of the robot (default: hmm)')
    parser.add_argument('--user', choices=sorted(config.SIM_USERS), default='mixed',
                        help='simulated user profile (default: mixed)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--noise', type=float, default=1.0,
                        help='measurement noise multiplier (default: 1.0)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='simulated camera frame rate (default: 10)')
    parser.add_argument('--every-step', action='store_true',
                        help='print every time step, not only the ones where something happens')
    parser.add_argument('--no-plot', action='store_true', help='do not make the timeline figure')
    parser.add_argument('--show', action='store_true', help='also open the figure in a window')
    return parser.parse_args()


def fmt(value, digits=2):
    return '-' if value is None or value == '' else f'{value:.{digits}f}'


class StepPrinter:
    """Prints the time steps where the hidden state, flow state, events or action change."""

    def __init__(self, every_step):
        self.every_step = every_step
        self.previous = None

    def __call__(self, row):
        notes = []
        if self.previous is None or row['user_state'] != self.previous['user_state']:
            notes.append(f'USER is {row["user_state"]}')
        if row['events']:
            notes.append(f'event {row["events"]}')
        if row['action']:
            notes.append(f'ROBOT {row["action"]}: "{row["message"]}"')
        self.previous = row
        if not notes and not self.every_step:
            return
        belief = ''
        if row['p_engaged'] is not None:
            belief = (f'belief E/M/U {fmt(row["p_engaged"])}/{fmt(row["p_unmotivated"])}/'
                      f'{fmt(row["p_unstable"])}  ')
        print(f'{row["step"]:5d} {row["t"]:7.1f}s  {row["flow_state"]:9} '
              f'{row["exercise"] or "-":18} {row["progress"] or "":20} '
              f'balance {row["balance"] or "-":8}  {belief}{"; ".join(notes)}')


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else ('' if v is None else v))
                             for k, v in row.items()})


def plot_timeline(rows, title, path, show):
    """Four panels over time: hidden state + belief, balance level, movement, robot actions."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = [r['t'] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1.2, 1.5, 2.0]})
    for ax in axes:
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK, labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)

    def spans(key, value):
        """(start, end) times where row[key] == value."""
        result, start = [], None
        for r in rows:
            if r[key] == value and start is None:
                start = r['t']
            elif r[key] != value and start is not None:
                result.append((start, r['t']))
                start = None
        if start is not None:
            result.append((start, rows[-1]['t']))
        return result

    # (a) Hidden user state as background bands, HMM belief as lines
    ax = axes[0]
    for state, color in STATE_COLORS.items():
        for start, end in spans('user_state', state):
            ax.axvspan(start, end, color=color, alpha=0.16, linewidth=0)
    handles = [Patch(color=c, alpha=0.3, label=f'true: {s}') for s, c in STATE_COLORS.items()]
    if rows[0]['p_engaged'] is not None:
        for state, color in STATE_COLORS.items():
            line, = ax.plot(t, [r[f'p_{state}'] for r in rows], color=color, linewidth=1.8,
                            label=f'P({state})')
            handles.append(line)
        ax.set_ylabel('belief', color=INK)
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.set_yticks([])
        ax.text(0.01, 0.5, 'rule policy: no belief, only the true state is shown',
                transform=ax.transAxes, color=INK, fontsize=9, va='center')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8,
              frameon=False)
    ax.set_title('(a) Hidden user state (bands) and the robot\'s HMM belief (lines)',
                 loc='left', fontsize=10, color=INK)

    # (b) Balance level of each frame (what the camera suggests) and the sustained level
    ax = axes[1]
    levels = {'low': 0, 'moderate': 1, 'high': 2}
    raw = [(r['t'], levels[r['balance']]) for r in rows if r['balance']]
    sustained = [(r['t'], levels[r['balance_sustained']]) for r in rows if r['balance_sustained']]
    if sustained:
        ax.plot([p[0] for p in sustained], [p[1] for p in sustained], color='#c3c2b7',
                linewidth=3, label='sustained (rule policy input)', drawstyle='steps-post',
                zorder=1)
    ax.scatter([p[0] for p in raw], [p[1] for p in raw], s=6, color='#0b0b0b',
               label='this frame (HMM input)', linewidths=0, zorder=2)
    ax.set_yticks([0, 1, 2], ['low', 'moderate', 'high'])
    ax.set_ylim(-0.4, 2.4)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title('(b) Measured instability level', loc='left', fontsize=10, color=INK)

    # (c) Movement: hip drop (sit-to-stand) and foot lift (single-leg balance)
    ax = axes[2]
    ax.plot(t, [r['hip_drop'] if r['hip_drop'] is not None else float('nan') for r in rows],
            color='#0b0b0b', linewidth=1.2, label='hip drop')
    ax.plot(t, [abs(r['foot_lift']) if r['foot_lift'] is not None else float('nan') for r in rows],
            color='#8a8984', linewidth=1.2, label='foot lift')
    ax.set_ylabel('leg lengths', color=INK)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title('(c) Simulated movement as measured', loc='left', fontsize=10, color=INK)

    # (d) Robot actions, with the exercise blocks in the background
    ax = axes[3]
    names = ['INSTRUCT', 'PRAISE', 'MOTIVATE', 'REASSURE_SLOW', 'REASSURE_HOLD',
             'SIMPLIFY', 'REST', 'TRACKING_LOST']
    for start, end in spans('flow_state', 'doing'):
        row = next(r for r in rows if r['t'] >= start)
        ax.axvspan(start, end, color=GRID, alpha=0.6, linewidth=0)
        label = row['exercise'].replace('_', ' ') + (' (simplified)' if row['simplified'] else '')
        ax.text(start + 1, len(names) - 0.3, label, fontsize=8, color=INK, va='top')
    acted = [(r['t'], names.index(r['action'])) for r in rows if r['action']]
    ax.scatter([a[0] for a in acted], [a[1] for a in acted], s=40, color='#0b0b0b', zorder=3)
    ax.set_yticks(range(len(names)), [n.lower() for n in names])
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.set_xlabel('time (s)', color=INK)
    ax.set_title('(d) Robot actions (grey blocks: exercising)', loc='left', fontsize=10, color=INK)

    fig.suptitle(title, x=0.01, ha='left', fontsize=12, color='#0b0b0b')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    name = f'sim_{args.policy}_{args.user}_seed{args.seed}'
    print(f'Simulated session: policy={args.policy} user={args.user} seed={args.seed} '
          f'noise={args.noise} fps={args.fps}')
    print('  step       t  flow      exercise           progress             '
          'what happened (E/M/U = engaged/unmotivated/unstable)')

    rows = run_session(args.policy, args.user, args.seed, args.noise, args.fps,
                       on_step=StepPrinter(args.every_step))

    csv_path = DATA_DIR / f'{name}.csv'
    write_csv(rows, csv_path)
    summary = summarize(rows)
    print(f'\n{len(rows)} time steps, log: {csv_path}')
    for key, value in summary.items():
        print(f'  {key:22} {fmt(value) if isinstance(value, float) else value}')

    if not args.no_plot:
        png_path = DATA_DIR / f'{name}.png'
        title = (f'Simulated session - {args.policy} policy, {args.user} user, '
                 f'seed {args.seed}, noise {args.noise}')
        plot_timeline(rows, title, png_path, args.show)
        print(f'  figure: {png_path}')


if __name__ == '__main__':
    main()
