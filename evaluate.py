"""Compare the rule policy and the HMM belief policy in simulation (assignment 3).

Runs many simulated sessions for each policy:
- with the 'mixed' user at three noise levels (how robust is each policy to noise?)
- with every user profile at normal noise (does the robot adapt to different users?)
The same seeds are used for both policies. The sessions still differ afterwards,
because the robot's actions change what the simulated user does.

Writes data/evaluation.csv (one row per session) and data/evaluation.png, and prints
a table with the mean (and standard deviation) of each metric. The metrics are
described in coach/Simulation.summarize().

    python evaluate.py              # 20 seeds per condition
    python evaluate.py --seeds 50
"""

import argparse
import csv
import statistics
from pathlib import Path

from coach import config
from coach.Simulation import run_session, summarize

DATA_DIR = Path(__file__).parent / 'data'
POLICIES = ('rule', 'hmm')
NOISE_LEVELS = (0.5, 1.0, 2.0)
# Two series: first two slots of the validated categorical palette
POLICY_COLORS = {'rule': '#2a78d6', 'hmm': '#eb6834'}
POLICY_LABELS = {'rule': 'rule policy', 'hmm': 'HMM belief policy'}
INK = '#52514e'
GRID = '#e4e3df'

# Metrics in the printed table: (key, header, digits)
TABLE = [
    ('detected', 'detected', 2),
    ('latency_s', 'latency s', 1),
    ('false_alarms', 'false alarms', 1),
    ('unstable_s', 'unstable s', 1),
    ('motivate_latency_s', 'motiv. lat. s', 1),
    ('simplified', 'simplify', 1),
    ('messages', 'messages', 1),
    ('duration_s', 'duration s', 0),
    ('belief_accuracy', 'belief acc.', 2),
    ('robot_helped', 'robot helped', 1),
]


def run_all(n_seeds):
    """:return: list of dicts: condition + summarize() of one session"""
    conditions = [('mixed', noise) for noise in NOISE_LEVELS]
    conditions += [(user, 1.0) for user in config.SIM_USERS if user != 'mixed']
    results = []
    for user, noise in conditions:
        for policy in POLICIES:
            for seed in range(n_seeds):
                summary = summarize(run_session(policy, user, seed, noise))
                results.append({'policy': policy, 'user': user, 'noise': noise, 'seed': seed,
                                **summary})
            print(f'  done: {policy:4} user={user:11} noise={noise}')
    return results


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


def print_table(results):
    header = f'{"user":11} {"noise":>5} {"policy":6}' + ''.join(f'{h:>16}' for _, h, _ in TABLE)
    print('\nMean (standard deviation) per session\n' + header)
    seen = []
    for r in results:
        key = (r['user'], r['noise'], r['policy'])
        if key in seen:
            continue
        seen.append(key)
        group = [x for x in results if (x['user'], x['noise'], x['policy']) == key]
        cells = ''
        for metric, _, digits in TABLE:
            mean, sd = mean_sd([x[metric] for x in group])
            text = '-' if mean is None else f'{mean:.{digits}f} ({sd:.{digits}f})'
            cells += f'{text:>16}'
        print(f'{key[0]:11} {key[1]:>5} {key[2]:6}{cells}')


def write_csv(results, path):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def plot(results, path):
    """Three safety metrics against the noise level, one line per policy ('mixed' user)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('detected', 'Unstable episodes detected', 'share of episodes'),
        ('latency_s', 'Time to first safety action', 'seconds'),
        ('false_alarms', 'Safety actions while not unstable', 'per session'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (metric, title, unit) in zip(axes, panels):
        for policy in POLICIES:
            means, sds = [], []
            for noise in NOISE_LEVELS:
                group = [r[metric] for r in results
                         if r['policy'] == policy and r['user'] == 'mixed' and r['noise'] == noise]
                mean, sd = mean_sd(group)
                means.append(mean)
                sds.append(sd)
            color = POLICY_COLORS[policy]
            # Small sideways offset so the error bars of the two policies do not overlap
            offset = -0.03 if policy == 'rule' else 0.03
            ax.errorbar([n + offset for n in NOISE_LEVELS], means, yerr=sds, color=color,
                        linewidth=2, marker='o',
                        markersize=6, capsize=3, label=POLICY_LABELS[policy])
        ax.set_title(title, loc='left', fontsize=10, color='#0b0b0b')
        ax.set_xlabel('measurement noise (x normal)', color=INK, fontsize=9)
        ax.set_ylabel(unit, color=INK, fontsize=9)
        ax.set_xticks(NOISE_LEVELS)
        ax.set_ylim(bottom=0)
        if metric == 'detected':
            ax.set_ylim(0, 1.05)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK, labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare rule and HMM policy in simulation.')
    parser.add_argument('--seeds', type=int, default=20,
                        help='sessions per condition (default: 20)')
    args = parser.parse_args()

    print(f'Running {args.seeds} sessions per policy and condition...')
    results = run_all(args.seeds)
    DATA_DIR.mkdir(exist_ok=True)
    write_csv(results, DATA_DIR / 'evaluation.csv')
    print_table(results)
    plot(results, DATA_DIR / 'evaluation.png')
    print(f'\nPer-session results: {DATA_DIR / "evaluation.csv"}')
    print(f'Figure: {DATA_DIR / "evaluation.png"}')


if __name__ == '__main__':
    main()
