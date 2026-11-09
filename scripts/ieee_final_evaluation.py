"""Frozen checkpoint contract, label-independent inference and final reporting."""
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import statistics

import numpy as np
import torch
from scipy.stats import t
from sklearn.metrics import average_precision_score

from ieee_multiseed import METHODS, SEEDS, SOURCE_FILES, ROOT, read, sha, write, verify_hashes
from run_ieee_no_replay_baseline import Network, BATCH_SIZE, EPOCHS, LEARNING_RATE
from ieee_final_data import validate_preparation, load_partition, load_preprocessor, transform_frame

CODE_FILES = (*SOURCE_FILES, 'scripts/ieee_final_data.py', 'scripts/ieee_final_evaluation.py',
              'scripts/run_ieee_final_evaluation.py', 'docs/ieee-final-evaluation-protocol.md')
METRICS = ('ap', 'fraud_recall_at_5pct', 'amount_weighted_fraud_recall_at_5pct')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_contract(preparation_path, windows, study_registry, source_root=ROOT):
    """Metadata/development artifacts only. Never open or hash the raw CSV."""
    preparation_path, windows, study_registry = map(lambda p: Path(p).resolve(), (preparation_path, windows, study_registry))
    prepared, verification, registry = read(preparation_path), read(windows/'verification.json'), read(study_registry)
    validate_preparation(prepared)
    if (registry['status'] != 'complete' or registry['seeds'] != list(SEEDS) or registry['methods'] != list(METHODS)
            or len(registry['runs']) != 25):
        raise ValueError('Incomplete or changed five-seed study')
    if (verification['source_sha256_from_preparation'] != prepared['source_sha256']
            or verification['preprocessing']['fit_scope'] != 'window 1 training subset only'
            or not verification['preprocessing']['frozen_for_later_windows']):
        raise ValueError('Training-only preprocessing provenance failed')
    expected = {str(windows/name) for name in ('verification.json', 'feature_names.json', 'preprocessor.joblib')}
    expected |= {str(windows/f'window-{w:02d}-{part}-{suffix}') for w in range(1, 7)
                 for part in ('training', 'held_out') for suffix in ('features.npz', 'metadata.csv')}
    data_hashes = registry['prepared_data']['files_sha256']
    if set(data_hashes) != expected:
        raise ValueError('Prepared-file manifest does not match fixed development paths')
    verify_hashes(data_hashes)
    for path, digest in registry['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Executed training source changed')
    names = read(windows/'feature_names.json')
    width = verification['preprocessing']['feature_count']
    if (len(names) != width or len(names) != len(set(names))
            or 'TransactionAmt' not in verification['predictors']['excluded_from_model_inputs']):
        raise ValueError('Wrong predictor schema')
    preprocessor = load_preprocessor(windows/'preprocessor.joblib', names)
    # Validate both fitted input and transformed names without any fitting.
    from prepare_ieee_windows import PREDICTORS, METADATA
    if (list(preprocessor.feature_names_in_) != PREDICTORS
            or any(any(key in name for key in METADATA) for name in names)):
        raise ValueError('Metadata enters classifier predictors')
    artifacts = dict(data_hashes)
    for path in (preparation_path, study_registry):
        artifacts[str(path)] = sha(path)
    models, seen, fixed_weight, versions = [], set(), None, None
    for record in registry['runs']:
        seed, method = record['seed'], record['method']
        if (seed, method) in seen or seed not in SEEDS or method not in METHODS:
            raise ValueError('Wrong or duplicate model pair')
        seen.add((seed, method))
        if record['status'] != ('reused' if seed == 501 else 'passed'):
            raise ValueError('Missing successful model artifact')
        directory = Path(record['path']).resolve()
        config_path, checkpoint_path = directory/'configuration.json', directory/'checkpoint-window-06.pt'
        config = read(config_path)
        weight = config.get('fixed_positive_weight', config.get('positive_weight'))
        if fixed_weight is None:
            fixed_weight, versions = weight, config['versions']
        if (config['seed'] != seed or config['input_width'] != width or config['device'] != 'cpu'
                or config['epochs_per_window'] != EPOCHS or config['learning_rate'] != LEARNING_RATE
                or config.get('batch_size', config.get('batch_size_current_examples')) != BATCH_SIZE
                or weight != fixed_weight or config['versions'] != versions
                or config.get('source_preparation_sha256', config.get('source_sha256_from_preparation')) != prepared['source_sha256']):
            raise ValueError('Frozen training configuration mismatch')
        if seed != 501 and (config['data_split_seed'] != 501 or config['training_seed'] != seed
                            or config['prepared_files_sha256'] != data_hashes):
            raise ValueError('Training/split seed or input hashes changed')
        if method != 'no-replay' and (
                config.get('policy', 'random') != ('combined' if method == 'combined-classnorm' else method)
                or config.get('alpha', .5) != .5 or config.get('combination', 'sum') != 'sum'
                or config.get('random_fraction', 0.) != 0.
                or config.get('score_normalization', 'global') != ('class' if method == 'combined-classnorm' else 'global')):
            raise ValueError('Policy or hyperparameter changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['after_training_window'] != 6 or checkpoint['seed'] != seed or checkpoint['input_width'] != width:
            raise ValueError('Wrong final checkpoint')
        model = Network(width)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite checkpoint')
        artifacts[str(config_path)], artifacts[str(checkpoint_path)] = sha(config_path), sha(checkpoint_path)
        models.append({'seed': seed, 'method': method, 'configuration': str(config_path), 'checkpoint': str(checkpoint_path)})
    if seen != {(seed, method) for seed in SEEDS for method in METHODS}:
        raise ValueError('Expected all 25 frozen models')
    if any(version(name) != value for name, value in versions.items()):
        raise ValueError('Installed package versions differ from development')
    contract = {'schema': 1, 'methods': list(METHODS), 'seeds': list(SEEDS), 'models': models,
                'preparation': prepared, 'source': str((Path(source_root)/prepared['source']).resolve()),
                'preprocessor': str(windows/'preprocessor.joblib'), 'feature_names': names,
                'artifacts_sha256': artifacts, 'implementation_sha256': {name: sha(ROOT/name) for name in CODE_FILES},
                'training_cutoff': prepared['partitions']['development']['max_transaction_dt'],
                'settings': {'batch_size': BATCH_SIZE, 'epochs_per_window': EPOCHS, 'learning_rate': LEARNING_RATE,
                             'positive_weight': fixed_weight, 'review_fraction': .05, 'memory_per_class': 128,
                             'retrieve_per_class': 64, 'alpha': .5, 'combination': 'sum', 'random_fraction': 0.,
                             'updates_after_development': 0}, 'versions': versions}
    if contract['source'] in artifacts:
        raise ValueError('Raw source must not be opened by contract checks')
    contract['contract_sha256'] = identity(contract)
    return contract


def verify_contract(contract):
    unsigned = {key: value for key, value in contract.items() if key != 'contract_sha256'}
    if identity(unsigned) != contract['contract_sha256']:
        raise ValueError('Contract digest mismatch')
    if contract['methods'] != list(METHODS) or contract['seeds'] != list(SEEDS):
        raise ValueError('Frozen method/seed set changed')
    verify_hashes(contract['artifacts_sha256'])
    if contract['implementation_sha256'] != {name: sha(ROOT/name) for name in CODE_FILES}:
        raise ValueError('Implementation/protocol changed since freeze')
    if any(version(name) != value for name, value in contract['versions'].items()):
        raise ValueError('Package versions changed since freeze')


def predict(model, matrix):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, matrix.shape[0], BATCH_SIZE):
            features = torch.from_numpy(np.asarray(matrix[start:start+BATCH_SIZE].toarray(), dtype=np.float32))
            scores.append(torch.sigmoid(model(features)).numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError('Invalid predicted score')
    return result


def score_predictions(scores, metadata):
    scores = np.asarray(scores)
    raw_ids = metadata['TransactionID'].to_numpy()
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError('Transaction IDs must be exact integers')
    ids = raw_ids.astype(np.int64)
    labels = metadata['isFraud'].to_numpy()
    amounts = metadata['TransactionAmt'].to_numpy(dtype=np.float64)
    if (len(scores) != len(ids) or scores.ndim != 1 or len(ids) == 0 or len(set(ids)) != len(ids)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(amounts).all() or (amounts < 0).any()
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Invalid prediction/scoring metadata')
    top = np.lexsort((ids, -scores))[:math.ceil(.05*len(ids))]
    positives = int(labels.sum())
    total_amount = float(amounts[labels == 1].sum())
    if not math.isfinite(total_amount):
        raise ValueError('Nonfinite fraud amount total')
    return {'rows': len(ids), 'fraud_count': positives, 'review_quota_rows': len(top),
            'ap': float(average_precision_score(labels, scores)) if positives else None,
            'fraud_recall_at_5pct': float(labels[top].sum()/positives) if positives else None,
            'amount_weighted_fraud_recall_at_5pct': float(amounts[top][labels[top] == 1].sum()/total_amount) if total_amount > 0 else None}


def aggregate_metrics(rows):
    if len(rows) != 25 or {(r['seed'], r['method']) for r in rows} != {(s, m) for s in SEEDS for m in METHODS}:
        raise ValueError('Incomplete final comparison')
    summary = []
    for method in METHODS:
        entry = {'method': method}
        for metric in METRICS:
            values = [r[metric] for r in rows if r['method'] == method]
            entry[metric] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values)} if all(v is not None for v in values) else {'mean': None, 'sd': None}
        summary.append(entry)
    lookup = {(r['seed'], r['method']): r for r in rows}
    paired = [{'seed': seed, **{metric: (lookup[seed, 'combined-classnorm'][metric]-lookup[seed, 'combined'][metric])
               if lookup[seed, 'combined-classnorm'][metric] is not None and lookup[seed, 'combined'][metric] is not None else None
               for metric in METRICS}} for seed in SEEDS]
    contrasts = {}
    for metric in METRICS:
        values = [r[metric] for r in paired]
        if any(v is None for v in values):
            contrasts[metric] = {'available': False}
            continue
        mean, sd = statistics.mean(values), statistics.stdev(values)
        halfwidth = float(t.ppf(.975, 4))*sd/math.sqrt(5)
        contrasts[metric] = {'available': True, 'mean': mean, 'sd': sd,
            'paired_t_95_interval': [mean-halfwidth, mean+halfwidth],
            'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values), 'ties': sum(v == 0 for v in values)}
    return {'per_seed': rows, 'summary': summary, 'paired_differences': paired, 'contrasts': contrasts,
            'primary_metric': 'amount_weighted_fraud_recall_at_5pct',
            'primary_direction': 'class-normalized combined minus original combined',
            'uncertainty': 'Sample SD and exploratory paired t intervals across training seeds on one fixed temporal partition'}


def verify_validation_receipt(path, contract):
    path = Path(path).resolve()
    receipt = read(path)
    if (receipt['status'] != 'complete' or receipt['phase'] != 'validation'
            or receipt['contract_sha256'] != contract['contract_sha256']
            or not receipt['all_predictions_saved_before_metrics']):
        raise ValueError('Reserved requires completed validation for the same contract')
    data = receipt['data']
    parts = contract['preparation']['partitions']
    prefix = data.get('validation_prefix_sha256')
    if (receipt.get('updates_after_development') != 0 or data.get('phase') != 'validation'
            or data.get('rows') != parts['validation']['row_count']
            or data.get('consumed_rows') != parts['development']['row_count']+parts['validation']['row_count']
            or data.get('full_source_verified') is not False
            or not isinstance(prefix, str) or len(prefix) != 64
            or any(c not in '0123456789abcdef' for c in prefix)
            or data.get('consumed_source_sha256') != prefix):
        raise ValueError('Invalid validation data lineage')
    expected = {f'predictions-{seed}-{method}.npz' for seed in SEEDS for method in METHODS} | {'metrics.json'}
    if set(receipt['files_sha256']) != expected:
        raise ValueError('Validation receipt lacks required prediction/metric files')
    verify_hashes({str(path.parent/name): value for name, value in receipt['files_sha256'].items()})
    return receipt


def execute_phase(contract, phase, output, validation_receipt=None):
    if phase not in ('validation', 'reserved'):
        raise ValueError('Invalid evaluation phase')
    verify_contract(contract)
    validation = None
    if phase == 'reserved':
        if validation_receipt is None:
            raise ValueError('Reserved requires a validation receipt before source access')
        validation = verify_validation_receipt(validation_receipt, contract)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output/'status.json', {'status': 'started', 'phase': phase, 'contract_sha256': contract['contract_sha256']})
    frame, lineage = load_partition(contract['source'], contract['preparation'], phase,
                                    expected_prefix=validation['data']['validation_prefix_sha256'] if validation else None)
    preprocessor = load_preprocessor(contract['preprocessor'], contract['feature_names'])
    matrix = transform_frame(frame, preprocessor, contract['feature_names'])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    files, predictions = {}, []
    # Persist EVERY model's predictions before any outcome-based scoring.
    for record in contract['models']:
        seed, method = record['seed'], record['method']
        checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=True)
        model = Network(matrix.shape[1])
        model.load_state_dict(checkpoint['model_state'], strict=True)
        scores = predict(model, matrix)
        name = f'predictions-{seed}-{method}.npz'
        np.savez_compressed(output/name, transaction_id=frame.TransactionID.to_numpy(dtype=np.int64), scores=scores)
        files[name] = sha(output/name)
        predictions.append((record, scores))
    write(output/'predictions-complete.json', {'phase': phase, 'contract_sha256': contract['contract_sha256'], 'files_sha256': files})
    rows = [{'seed': record['seed'], 'method': record['method'], **score_predictions(scores, frame)} for record, scores in predictions]
    metrics = aggregate_metrics(rows)
    metrics.update(phase=phase, contract_sha256=contract['contract_sha256'])
    write(output/'metrics.json', metrics)
    files['metrics.json'] = sha(output/'metrics.json')
    # Fail rather than issue a completion receipt if frozen artifacts drifted.
    verify_contract(contract)
    receipt = {'status': 'complete', 'phase': phase, 'contract_sha256': contract['contract_sha256'],
               'data': lineage, 'files_sha256': files, 'all_predictions_saved_before_metrics': True,
               'updates_after_development': 0}
    if validation_receipt is not None:
        receipt['validation_receipt_sha256'] = sha(validation_receipt)
    write(output/'receipt.json', receipt)
    write(output/'status.json', {'status': 'complete', 'phase': phase, 'contract_sha256': contract['contract_sha256']})
    return metrics
