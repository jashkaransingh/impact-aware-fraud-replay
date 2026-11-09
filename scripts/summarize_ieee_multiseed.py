"""Read-only paired audits and paper artifacts for the fixed five-seed study."""
import argparse
import csv
import math
import platform
import sys
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import t

from ieee_multiseed import (ROOT, SEEDS, METHODS, METRICS, SOURCE_FILES, audit_seed,
                           read, sha, verify_hashes, write, baseline)
from compare_ieee_policies import summarize

LABELS = {'no-replay': 'No replay', 'random': 'Random', 'forgetting': 'Forgetting',
          'combined': 'Combined', 'combined-classnorm': 'Class norm'}
TITLES = {'historical_ap': 'Historical AP', 'current_window_ap': 'Current AP (window 6)',
          'forgetting': 'AP forgetting (lower better)',
          'historical_amount_recall_5pct': 'Historical amount recall at 5%',
          'current_amount_recall_5pct': 'Current amount recall at 5%'}


def aggregate(rows):
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    if len(rows) != len(expected) or {(r['seed'], r['method']) for r in rows} != expected:
        raise ValueError('Expected exactly one result for each of the 25 prespecified pairs')
    if any(not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1 for row in rows for metric in METRICS):
        raise ValueError('Missing, nonfinite or out-of-range metric')
    lookup = {(r['seed'], r['method']): r for r in rows}
    summaries = []
    for method in METHODS:
        result = {'method': method, 'n': len(SEEDS)}
        for metric in METRICS:
            values = [lookup[seed, method][metric] for seed in SEEDS]
            result[metric] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values)}
        summaries.append(result)
    paired = [{'seed': seed, **{metric: lookup[seed, 'combined-classnorm'][metric] -
                               lookup[seed, 'combined'][metric] for metric in METRICS}}
              for seed in SEEDS]
    differences = {}
    for metric in METRICS:
        values = [r[metric] for r in paired]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        halfwidth = float(t.ppf(.975, len(values)-1))*sd/math.sqrt(len(values))
        differences[metric] = {'mean': mean, 'sd': sd, 'paired_t_95_interval': [mean-halfwidth, mean+halfwidth],
            'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values),
            'ties': sum(v == 0 for v in values),
            'improvements': sum(v < 0 if metric == 'forgetting' else v > 0 for v in values),
            'new_seeds_502_to_505_mean': statistics.mean(values[1:])}
    return {'summary': summaries, 'per_seed': rows, 'paired_differences': paired,
            'paired_summary': differences, 'sd_definition': 'sample SD across training seeds (ddof=1)',
            'interval_caveat': 'Exploratory paired t interval across training seeds on one fixed development dataset; seed 501 informed candidate selection. Not independent final evaluation.'}


def save_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for metric, ax in zip(METRICS, axes.flat):
        for index, summary in enumerate(report['summary']):
            method = summary['method']
            values = [r[metric] for r in report['per_seed'] if r['method'] == method]
            ax.scatter(index+np.linspace(-.13, .13, len(values)), values, s=18, color='#5892b9', alpha=.8)
            ax.errorbar(index, summary[metric]['mean'], yerr=summary[metric]['sd'],
                        fmt='o', color='#182e41', capsize=4, markersize=5)
        ax.set(title=TITLES[metric], xticks=range(len(METHODS)), xticklabels=[LABELS[m] for m in METHODS])
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.grid(axis='y', alpha=.25)
    axes[1, 2].axis('off')
    axes[1, 2].text(.05, .9, 'Fixed development windows\nTraining seeds 501–505\n\nPoints: individual seeds\nBlack: mean ± sample SD\n\nHistorical: final windows 1–5\nCurrent: final window 6\n\nEach panel uses its own scale.', va='top')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'method-summary.{extension}', dpi=180,
                    metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
    plt.close(fig)
    chosen = ('historical_amount_recall_5pct', 'historical_ap', 'forgetting')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for metric, ax in zip(chosen, axes):
        values = [r[metric]*100 for r in report['paired_differences']]
        ax.bar(range(5), values, color=['#337ca0' if v >= 0 else '#ce6a35' for v in values])
        mean = report['paired_summary'][metric]['mean']*100
        ax.axhline(0, color='#333333', linewidth=.8)
        ax.axhline(mean, color='#555555', linestyle='--', label=f'Mean {mean:+.3f}')
        ax.set(title=TITLES[metric], xticks=range(5), xticklabels=SEEDS, xlabel='Training seed',
               ylabel='Class norm − combined (×100)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'paired-differences.{extension}', dpi=180,
                    metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing existing report directory: {args.output}')
    registry = read(args.study/'registry.json')
    if (registry['status'] != 'complete' or registry['seeds'] != list(SEEDS) or registry['methods'] != list(METHODS)
            or len(registry['runs']) != 25 or any(r['status'] not in ('reused', 'passed') for r in registry['runs'])):
        raise AssertionError('Incomplete or changed study')
    if sha(ROOT/'docs/ieee-multiseed-protocol.md') != registry['protocol_sha256']:
        raise AssertionError('Prespecified protocol changed after launch')
    feature_names = read(baseline.INPUT/'feature_names.json')
    if len(feature_names) != registry['prepared_data']['input_width'] or any('TransactionAmt' in name for name in feature_names):
        raise AssertionError('Transformed feature names include amount or have wrong width')
    verify_hashes(registry['prepared_data']['files_sha256'])
    verify_hashes(registry['protected_existing_artifacts'])
    if registry['source_sha256'] != {name: sha(ROOT/name) for name in SOURCE_FILES}:
        raise AssertionError('Executed training code changed')
    audits, rows, provenance = {}, [], {}
    for seed in SEEDS:
        paths = {r['method']: Path(r['path']) for r in registry['runs'] if r['seed'] == seed}
        audits[str(seed)] = audit_seed(paths, seed, registry['prepared_data'])
        for method in METHODS:
            path = paths[method]
            rows.append({'seed': seed, 'method': method, **{key: value for key, value in
                         summarize(read(path/'results.json')).items() if key in METRICS}})
            provenance[f'{seed}/{method}'] = {'path': str(path), 'results_sha256': sha(path/'results.json'),
                                            'configuration_sha256': sha(path/'configuration.json')}
    report = aggregate(rows)
    report.update(audits=audits, provenance=provenance, training_seeds=list(SEEDS), data_split_seed=501,
                  training_source_sha256=registry['source_sha256'],
                  prepared_files_sha256=registry['prepared_data']['files_sha256'],
                  protocol_sha256=registry['protocol_sha256'],
                  reporting_source_sha256=sha(Path(__file__)),
                  runtime_environment={'python': sys.version, 'platform': platform.platform()},
                  transformed_features_exclude_amount=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'summary.json', report)
    flat = [{'method': r['method'], 'n': r['n'], **{f'{metric}_{stat}': r[metric][stat]
             for metric in METRICS for stat in ('mean', 'sd')}} for r in report['summary']]
    save_csv(args.output/'method-summary.csv', flat)
    save_csv(args.output/'per-seed.csv', report['per_seed'])
    save_csv(args.output/'paired-differences.csv', report['paired_differences'])
    lines = ['Values are mean ± sample SD over seeds 501–505. Historical windows 1–5; current window 6. Amount recall at fixed 5% review quota.', '',
             '| Method | ' + ' | '.join(TITLES[m] for m in METRICS) + ' |',
             '| --- | ' + ' | '.join(['---']*len(METRICS)) + ' |']
    for row in report['summary']:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[m]["mean"]:.6f} ± {row[m]["sd"]:.6f}' for m in METRICS)+' |')
    lines += ['', 'Paired differences: class normalization minus original combined. Positive forgetting is worse.', '',
              '| Seed | '+' | '.join(TITLES[m] for m in METRICS)+' |',
              '| --- | '+' | '.join(['---']*len(METRICS))+' |']
    for row in report['paired_differences']:
        lines.append('| '+str(row['seed'])+' | '+' | '.join(f'{row[m]:+.6f}' for m in METRICS)+' |')
    lines += ['', report['interval_caveat']]
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    plots(report, args.output)
    print('\n'.join(lines), flush=True)
    print('Paired summary:', report['paired_summary'], flush=True)


if __name__ == '__main__':
    main()
