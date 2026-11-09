"""Audit and summarize development replay runs without reading reserved data.

All outputs go to a new directory. Inputs are aggregate results and replay logs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NO_REPLAY = ROOT / 'results/ieee-no-replay-baseline-501'
RANDOM = ROOT / 'results/ieee-class-balanced-replay-501'


def read(path):
    return json.loads(path.read_text())


def summarize(results):
    if len(results) != 6 or [r['after_training_window'] for r in results] != list(range(1, 7)):
        raise ValueError('Expected six complete chronological windows')
    trajectories = []
    for window in range(1, 7):
        trajectory = [next(m for m in step['held_out_evaluations']
                           if m['held_out_window'] == window)
                      for step in results[window-1:]]
        trajectories.append(trajectory)
    historical = [t[-1] for t in trajectories[:5]]
    mean = lambda values: sum(values) / len(values)
    return {
        'historical_ap': mean([m['ap'] for m in historical]),
        'current_window_ap': trajectories[5][-1]['ap'],
        'forgetting': mean([max(m['ap'] for m in t) - t[-1]['ap'] for t in trajectories[:5]]),
        'historical_amount_recall_5pct': mean([m['amount_weighted_fraud_recall_at_5pct'] for m in historical]),
        'current_amount_recall_5pct': trajectories[5][-1]['amount_weighted_fraud_recall_at_5pct'],
        'historical_fraud_recall_5pct': mean([m['fraud_recall_at_5pct'] for m in historical]),
        'mean_diagonal_ap': mean([t[0]['ap'] for t in trajectories]),
    }


def per_window(results):
    rows = []
    for window in range(1, 7):
        trajectory = [next(m for m in step['held_out_evaluations'] if m['held_out_window'] == window)
                      for step in results[window-1:]]
        final = trajectory[-1]
        best = max(m['ap'] for m in trajectory)
        rows.append({'held_out_window': window, 'initial_ap': trajectory[0]['ap'],
                     'final_ap': final['ap'], 'best_ap': best, 'forgetting': best-final['ap'],
                     'final_amount_recall_5pct': final['amount_weighted_fraud_recall_at_5pct'],
                     'final_fraud_recall_5pct': final['fraud_recall_at_5pct']})
    return rows


def plot_windows(windows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, rows in windows.items():
        for ax, key in zip(axes, ('final_ap', 'final_amount_recall_5pct')):
            ax.plot(range(1, 7), [r[key] for r in rows], marker='o', label=name)
    for ax, title in zip(axes, ('Final AP', 'Final fraud amount recall at 5% review')):
        ax.set(title=title, xlabel='Held-out development window', xticks=range(1, 7))
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'final-performance.png', dpi=160)
    plt.close(fig)


def verify_random(candidate):
    checks = {name: read(candidate / name) == read(RANDOM / name)
              for name in ('results.json', 'replay-ids.json', 'smoke-check.json')}
    old, new = read(RANDOM / 'configuration.json'), read(candidate / 'configuration.json')
    checks['versions'] = old['versions'] == new['versions']
    for key in ('seed', 'input_width', 'positive_weight', 'batch_size_current_examples',
                'epochs_per_window', 'learning_rate', 'source_sha256_from_preparation'):
        checks[key] = old[key] == new[key]
    if not all(checks.values()):
        raise AssertionError(f'Random reproduction failed: {checks}')
    return checks


def audit_run(path, baseline_results, anchor_log):
    results, log = read(path / 'results.json'), read(path / 'replay-ids.json')
    if len(log['retrievals']) != 18:
        raise AssertionError('Missing retrieval epochs')
    if log['memory_snapshots'] != anchor_log['memory_snapshots']:
        raise AssertionError('Reservoir admission changed across policies')
    if results[0]['held_out_evaluations'] != baseline_results[0]['held_out_evaluations']:
        raise AssertionError('Window-1 evaluation differs')
    if results[0]['epoch_mean_losses'] != baseline_results[0]['epoch_mean_losses']:
        raise AssertionError('Window-1 training differs')
    # Runner verification checks full metadata membership and exclusion; also
    # check saved invariants and retrieval exposure against the random anchor.
    verification = read(path / 'verification.json')
    if not all(v for v in verification.values() if isinstance(v, bool)):
        raise AssertionError('Runner verification failed')
    for result, base in zip(results, baseline_results):
        if result['training_rows'] != base['training_rows']:
            raise AssertionError('Current exposure differs')
    for record, reference in zip(log['retrievals'], anchor_log['retrievals']):
        window = record['training_window']
        pools = ({'0': [], '1': []} if window == 1 else
                 log['memory_snapshots'][window-2]['slots_by_class'])
        prior = {item['transaction_id']: (int(label), item['source_window'])
                 for label, pool in pools.items() for item in pool}
        ids = [item['transaction_id'] for item in record['retrieved']]
        if len(ids) != len(set(ids)):
            raise AssertionError('Duplicate retrieval ID')
        for item in record['retrieved']:
            if prior.get(item['transaction_id']) != (item['class'], item['source_window']):
                raise AssertionError('Retrieved ID is outside prior reservoir or has wrong class/source')
        for label in (0, 1):
            count = sum(item['class'] == label for item in record['retrieved'])
            if count != min(64, len(pools[str(label)])):
                raise AssertionError('Class retrieval quota changed')
        for key in ('training_window', 'epoch', 'current_rows_exposed', 'optimizer_steps'):
            if record[key] != reference[key]:
                raise AssertionError(f'{key} differs')
        if len(record['retrieved']) != len(reference['retrieved']):
            raise AssertionError('Replay budget differs')
    config, anchor = read(path / 'configuration.json'), read(RANDOM / 'configuration.json')
    for key in ('seed', 'input_width', 'positive_weight', 'batch_size_current_examples',
                'epochs_per_window', 'learning_rate', 'source_sha256_from_preparation', 'versions'):
        if config[key] != anchor[key]:
            raise AssertionError(f'Configuration mismatch: {key}')
    return {'same_reservoir_ids': True, 'same_exposure_steps_budget_versions': True,
            'runner_training_membership_checks_passed': True, 'window_1_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-reproduction', type=Path, required=True)
    parser.add_argument('--run', action='append', default=[], help='NAME=OUTPUT_DIR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reproduction = verify_random(args.random_reproduction)
    paths = {'no-replay': NO_REPLAY, 'random': RANDOM}
    for entry in args.run:
        name, path = entry.split('=', 1)
        if name in paths:
            raise ValueError(f'Duplicate method: {name}')
        paths[name] = Path(path)
    base, anchor = read(NO_REPLAY / 'results.json'), read(RANDOM / 'replay-ids.json')
    audits = {'random-reproduction': reproduction}
    audits['random-reproduction-run'] = audit_run(args.random_reproduction, base, anchor)
    rows = []
    provenance = {}
    windows = {}
    for name, path in paths.items():
        if name not in ('no-replay', 'random'):
            audits[name] = audit_run(path, base, anchor)
        method_results = read(path / 'results.json')
        rows.append({'method': name, **summarize(method_results)})
        windows[name] = per_window(method_results)
        provenance[name] = {'path': str(path.resolve()),
                            'results_sha256': hashlib.sha256((path / 'results.json').read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'metrics_definition': 'Final mean historical windows 1-5; current window 6; forgetting mean best-to-final AP on 1-5; amount recall at fixed 5% quota. Unweighted window means.',
              'rows': rows, 'audits': audits, 'provenance': provenance, 'per_window': windows}
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    columns = ['method', 'historical_ap', 'current_window_ap', 'forgetting',
               'historical_amount_recall_5pct', 'current_amount_recall_5pct']
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(str(row[k]) if k == 'method' else f'{row[k]:.6f}' for k in columns) + ' |' for row in rows]
    (args.output / 'comparison.md').write_text(report['metrics_definition'] + '\n\n' + '\n'.join(lines) + '\n')
    plot_windows(windows, args.output)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
