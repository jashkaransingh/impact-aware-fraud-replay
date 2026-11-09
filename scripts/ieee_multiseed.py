"""Fixed paired-study adapter and read-only audits for the legacy IEEE runners.

The original runner source files and all seed-501 artifacts remain unchanged.
Only training constants vary; prepared input/split paths never depend on seed.
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_ieee_no_replay_baseline as baseline
import run_ieee_balanced_replay as replay
import compare_ieee_policies as comparison

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (501, 502, 503, 504, 505)
SPLIT_SEED = 501
METHODS = ('no-replay', 'random', 'forgetting', 'combined', 'combined-classnorm')
METRICS = ('historical_ap', 'current_window_ap', 'forgetting',
           'historical_amount_recall_5pct', 'current_amount_recall_5pct')
REUSED = {
    'no-replay': ROOT / 'results/ieee-no-replay-baseline-501',
    'random': ROOT / 'results/ieee-class-balanced-replay-501',
    'forgetting': ROOT / 'results/ieee-policy-study-501/forgetting',
    'combined': ROOT / 'results/ieee-policy-study-501/combined',
    'combined-classnorm': ROOT / 'results/ieee-policy-study-501/candidates/combined-classnorm',
}
SOURCE_FILES = ('scripts/ieee_multiseed.py', 'scripts/run_ieee_multiseed.py',
                'scripts/run_ieee_no_replay_baseline.py', 'scripts/run_ieee_balanced_replay.py',
                'scripts/compare_ieee_policies.py', 'src/impact_replay/replay.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(paths):
    return {str(path): sha(path) for path in sorted(paths)}


def verify_hashes(manifest):
    for path, digest in manifest.items():
        if sha(path) != digest:
            raise AssertionError(f'Artifact changed: {path}')


def preparation_manifest(input_path=None):
    input_path = baseline.INPUT if input_path is None else Path(input_path)
    prepared = read(input_path / 'verification.json')
    if f'{SPLIT_SEED}:' not in prepared['holdout_rule']:
        raise AssertionError('Unexpected prepared split seed')
    predictors = prepared['predictors']
    if ('TransactionAmt' not in predictors['excluded_from_model_inputs']
            or 'TransactionAmt' in predictors['numeric'] + predictors['categorical']):
        raise AssertionError('Amount enters model features')
    if not all(prepared['verification'][k] for k in (
            'development_rows_assigned_once', 'training_and_held_out_ids_disjoint',
            'both_classes_in_each_subset')):
        raise AssertionError('Preparation verification failed')
    files = [input_path / name for name in ('verification.json', 'feature_names.json', 'preprocessor.joblib')]
    files += [input_path / f'window-{w:02d}-{part}-{kind}'
              for w in range(1, 7) for part in ('training', 'held_out')
              for kind in ('features.npz', 'metadata.csv')]
    return {'data_split_seed': SPLIT_SEED, 'files_sha256': hashes(files),
            'source_preparation_sha256': prepared['source_sha256_from_preparation'],
            'input_width': prepared['preprocessing']['feature_count']}


def trace_orders(seed, sizes):
    generator = torch.Generator().manual_seed(seed)
    records = []
    for window, size in enumerate(sizes, 1):
        for epoch in range(1, baseline.EPOCHS+1):
            order = torch.randperm(size, generator=generator)
            records.append({'window': window, 'epoch': epoch, 'rows': size,
                            'sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()})
    return records


def run_method(method, seed, output, paired_baseline=None, manifest=None):
    """Call unchanged training loops with scoped seed/output parameters.

    Configuration descriptions and added provenance are finalized only in the
    brand-new output. Original/reused artifacts are never edited.
    """
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Only the predetermined five methods/seeds are allowed')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    if method != 'no-replay' and paired_baseline is None:
        raise ValueError('Replay requires the same-seed no-replay run')
    if paired_baseline is not None and read(Path(paired_baseline)/'configuration.json')['seed'] != seed:
        raise ValueError('Baseline training seed mismatch')
    manifest = preparation_manifest() if manifest is None else manifest
    observed = []
    original_randperm = torch.randperm
    def observed_randperm(*args, **kwargs):
        order = original_randperm(*args, **kwargs)
        if kwargs.get('generator') is not None:
            index = len(observed)
            observed.append({'window': index//baseline.EPOCHS+1, 'epoch': index%baseline.EPOCHS+1,
                             'rows': len(order), 'sha256': hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        return order
    options = ['--policy', 'combined' if method == 'combined-classnorm' else method,
               '--output', str(output)]
    if method == 'combined-classnorm':
        options += ['--score-normalization', 'class']
    with patch.multiple(baseline, SEED=seed, OUTPUT=output), \
         patch.multiple(replay, SEED=seed, OUTPUT=output,
                        BASELINE=Path(paired_baseline) if paired_baseline is not None else replay.BASELINE), \
         patch.object(sys, 'argv', [str(replay.__file__), *options]), \
         patch.object(torch, 'randperm', side_effect=observed_randperm):
        if method == 'no-replay':
            baseline.main()
        else:
            replay.main()
    results = read(output/'results.json')
    expected = trace_orders(seed, [r['training_rows'] for r in results])
    if observed != expected:
        raise AssertionError('Observed current-data order differs from declared seed')
    config = read(output/'configuration.json')
    config.update(training_seed=seed, data_split_seed=SPLIT_SEED, study_method=method,
                  prepared_files_sha256=manifest['files_sha256'],
                  training_source_sha256={name: sha(ROOT/name) for name in SOURCE_FILES})
    if method != 'no-replay':
        config['random_generators'] = {
            'current_order': f'torch.Generator seed {seed}, matching same-seed baseline',
            'memory_admission': f'independent random.Random seed {seed}',
            'retrieval': f'independent random.Random seed {seed}'}
    write(output/'configuration.json', config)
    write(output/'current-order-hashes.json', observed)
    write(output/'adapter-verification.json', {'observed_current_order_matches_seed': True,
          'training_seed': seed, 'data_split_seed': SPLIT_SEED,
          'legacy_runners_unchanged': True})


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def audit_seed(paths, seed, manifest):
    """Recheck raw development membership and saved states without training."""
    if set(paths) != set(METHODS):
        raise AssertionError('Expected exactly five paired methods')
    paths = {m: Path(p) for m, p in paths.items()}
    prepared = read(baseline.INPUT/'verification.json')
    training = {w: pd.read_csv(baseline.INPUT/f'window-{w:02d}-training-metadata.csv') for w in range(1, 7)}
    held = {w: pd.read_csv(baseline.INPUT/f'window-{w:02d}-held_out-metadata.csv') for w in range(1, 7)}
    train_ids = {w: set(rows.TransactionID.astype(int)) for w, rows in training.items()}
    held_ids = set(i for rows in held.values() for i in rows.TransactionID.astype(int))
    seen = set()
    for w, ids in train_ids.items():
        if ids & held_ids or ids & seen or len(ids) != len(training[w]):
            raise AssertionError('Training/held-out/chronological ID overlap')
        seen.update(ids)
    labels = {int(row.TransactionID): int(row.isFraud) for rows in training.values() for row in rows.itertuples()}
    base = read(paths['no-replay']/'results.json')
    anchor = read(paths['random']/'replay-ids.json')
    reference_config = read(REUSED['no-replay']/'configuration.json')
    base_checkpoint = torch.load(paths['no-replay']/'checkpoint-window-01.pt', weights_only=True)
    expected_orders = trace_orders(seed, [len(training[w]) for w in range(1, 7)])
    weight = (len(training[1])-training[1].isFraud.sum())/training[1].isFraud.sum()
    audits = {}
    for method, path in paths.items():
        config, results = read(path/'configuration.json'), read(path/'results.json')
        expected_policy = 'combined' if method == 'combined-classnorm' else method
        if config['seed'] != seed or config['device'] != 'cpu':
            raise AssertionError('Wrong training seed/device')
        for key, expected in [('input_width', manifest['input_width']), ('learning_rate', baseline.LEARNING_RATE),
                              ('epochs_per_window', baseline.EPOCHS), ('versions', reference_config['versions'])]:
            if config[key] != expected:
                raise AssertionError(f'Fixed setting changed: {method} {key}')
        if config.get('source_preparation_sha256', config.get('source_sha256_from_preparation')) != manifest['source_preparation_sha256']:
            raise AssertionError('Preparation source changed')
        if (config.get('batch_size', config.get('batch_size_current_examples')) != baseline.BATCH_SIZE
                or config.get('fixed_positive_weight', config.get('positive_weight')) != weight):
            raise AssertionError('Training batch/weight changed')
        if results[0]['epoch_mean_losses'] != base[0]['epoch_mean_losses'] or results[0]['held_out_evaluations'] != base[0]['held_out_evaluations']:
            raise AssertionError('Paired window-1 mismatch')
        comparison.summarize(results)  # Requires all six steps.
        cumulative_steps = 0
        for w, result in enumerate(results, 1):
            if result['training_rows'] != len(training[w]) or len(result['epoch_mean_losses']) != baseline.EPOCHS:
                raise AssertionError('Current exposure or epochs changed')
            if not np.isfinite(result['epoch_mean_losses']).all():
                raise AssertionError('Nonfinite losses')
            evaluations = result['held_out_evaluations']
            if [r['held_out_window'] for r in evaluations] != list(range(1, w+1)):
                raise AssertionError('Missing or extra held-out evaluation')
            for metric in evaluations:
                rows = held[metric['held_out_window']]
                if metric['rows'] != len(rows) or metric['fraud_count'] != int(rows.isFraud.sum()) or metric['review_quota_rows'] != math.ceil(.05*len(rows)):
                    raise AssertionError('Evaluation membership/quota changed')
            cumulative_steps += baseline.EPOCHS*math.ceil(len(training[w])/baseline.BATCH_SIZE)
            checkpoint = torch.load(path/f'checkpoint-window-{w:02d}.pt', weights_only=True)
            if checkpoint['seed'] != seed or checkpoint['input_width'] != manifest['input_width']:
                raise AssertionError('Checkpoint provenance mismatch')
            if any(float(state['step']) != cumulative_steps for state in checkpoint['optimizer_state']['state'].values()):
                raise AssertionError('Optimizer step budget changed')
            if w == 1 and not equal_state(checkpoint, base_checkpoint):
                raise AssertionError('Paired window-1 checkpoint mismatch')
        if seed != 501:
            if (config['training_seed'] != seed or config['data_split_seed'] != SPLIT_SEED
                    or config['prepared_files_sha256'] != manifest['files_sha256']
                    or config['training_source_sha256'] != {name: sha(ROOT/name) for name in SOURCE_FILES}
                    or read(path/'current-order-hashes.json') != expected_orders):
                raise AssertionError('Seed/order/input/source provenance changed')
        if method != 'no-replay':
            if (config.get('policy', 'random') != expected_policy
                    or config.get('alpha', .5) != .5 or config.get('combination', 'sum') != 'sum'
                    or config.get('random_fraction', 0.) != 0.
                    or config.get('score_normalization', 'global') != ('class' if method == 'combined-classnorm' else 'global')):
                raise AssertionError('Policy/hyperparameter changed')
            scale = max(float(np.quantile(np.log1p(training[1].TransactionAmt), .95)), 1e-6)
            if 'impact_scale' in config and config['impact_scale'] != scale:
                raise AssertionError('Impact scale does not use window-1 training')
            with patch.object(comparison, 'RANDOM', paths['random']):
                comparison.audit_run(path, base, anchor)
            log = read(path/'replay-ids.json')
            for snapshot in log['memory_snapshots']:
                ids = []
                for label, pool in snapshot['slots_by_class'].items():
                    if len(pool) > 128:
                        raise AssertionError('Memory class cap exceeded')
                    for row in pool:
                        i, source = row['transaction_id'], row['source_window']
                        if (source > snapshot['after_training_window'] or i not in train_ids[source]
                                or i in held_ids or labels[i] != int(label)):
                            raise AssertionError('Memory contains non-training or future record')
                        ids.append(i)
                if len(ids) != len(set(ids)):
                    raise AssertionError('Duplicate memory ID')
            for record in log['retrievals']:
                for row in record['retrieved']:
                    i, source = row['transaction_id'], row['source_window']
                    if source >= record['training_window'] or i not in train_ids[source] or i in held_ids or labels[i] != row['class']:
                        raise AssertionError('Replay leakage')
        audits[method] = {'fixed_settings_and_versions': True, 'complete_windows_evaluations': True,
                          'same_seed_window_1_checkpoint': True, 'optimizer_steps_match': True,
                          'training_only_memory_and_replay': True,
                          'current_order': 'observed hashes match' if seed != 501 else 'legacy source/reproduction verified; expected order reconstructed'}
    return audits
