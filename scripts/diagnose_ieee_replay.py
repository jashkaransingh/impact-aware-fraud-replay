"""Training-only selection diagnostics; never loads held-out or reserved data."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_ieee_balanced_replay import Network, StratifiedReservoir, historical_losses
from impact_replay.replay import normalize_positive


def diagnose(path):
    config = json.loads((path / 'configuration.json').read_text())
    log = json.loads((path / 'replay-ids.json').read_text())
    training = {}
    for window in range(1, 7):
        from run_ieee_no_replay_baseline import load_subset
        training[window] = load_subset(window, 'training', config['input_width'])
    rows = {}
    for window, (matrix, metadata) in training.items():
        for index, record in metadata.iterrows():
            rows[int(record.TransactionID)] = (window, index, int(record.isFraud), float(record.TransactionAmt))
    def materialize(snapshot):
        result = []
        for label in ('0', '1'):
            for record in snapshot['slots_by_class'][label]:
                transaction_id = record['transaction_id']
                window, index, y, amount = rows[transaction_id]
                assert window == record['source_window']
                result.append(dict(transaction_id=transaction_id, source_window=window,
                                   features=np.asarray(training[window][0].getrow(index).toarray().ravel(), dtype=np.float32),
                                   **{'class': y}, amount=amount))
        return result
    model = Network(config['input_width'])
    def checkpoint(window):
        state = torch.load(path / f'checkpoint-window-{window:02d}.pt', weights_only=True)
        model.load_state_dict(state['model_state'])
    reference = {}
    first_epoch_scores = []
    retrieval_benchmarks = []
    training_ids = {w: set(metadata['TransactionID'].astype(int)) for w, (_, metadata) in training.items()}
    for window, snapshot in enumerate(log['memory_snapshots'], 1):
        pool = materialize(snapshot)
        checkpoint(window)
        fresh = [r for r in pool if r['source_window'] == window]
        reference.update({r['transaction_id']: float(v) for r, v in zip(fresh, historical_losses(model, fresh))})
        if window == 6:
            continue
        delta = historical_losses(model, pool) - [reference[r['transaction_id']] for r in pool]
        labels = np.array([r['class'] for r in pool])
        norm = normalize_positive(delta)
        if config.get('score_normalization', 'global') == 'class':
            for label in (0, 1):
                norm[labels == label] = normalize_positive(delta[labels == label])
        impact = np.array([min(np.log1p(r['amount']) / config['impact_scale'], 1)*r['class'] for r in pool])
        memory = StratifiedReservoir(random.Random(501), random.Random(501))
        memory.policy = config['policy']
        memory.alpha = config.get('alpha', .5)
        memory.random_fraction = config.get('random_fraction', 0.)
        memory.score_normalization = config.get('score_normalization', 'global')
        for row, value in zip(pool, impact):
            row['reference_loss'] = reference[row['transaction_id']]
            row['impact'] = value
        memory.slots = {label: [r for r in pool if r['class'] == label] for label in (0, 1)}
        # A training-only microbenchmark, not a new training experiment. The
        # empty held-out set suffices here: every candidate was materialized
        # from training matrices and separately audited against all held-out IDs.
        memory.retrieve(window+1, training_ids, set(), model=model)
        start = time.perf_counter()
        for _ in range(20):
            memory.retrieve(window+1, training_ids, set(), model=model)
        retrieval_benchmarks.append((time.perf_counter()-start)/20)
        first_epoch_scores.append({'before_window': window+1,
            'configured_normalization': memory.score_normalization,
            'positive_delta_p95_all': float(np.quantile(np.maximum(delta, 0), .95)),
            'positive_delta_p95_fraud': float(np.quantile(np.maximum(delta[labels == 1], 0), .95)),
            'positive_delta_p95_legitimate': float(np.quantile(np.maximum(delta[labels == 0], 0), .95)),
            'fraud_zero_forgetting_fraction': float(np.mean(delta[labels == 1] <= 0)),
            'fraud_normalized_forgetting_mean': float(np.mean(norm[labels == 1])),
            'fraud_impact_mean': float(np.mean(impact[labels == 1])),
            'fraud_impact_clipped_fraction': float(np.mean(impact[labels == 1] >= 1))})
    epochs = [r for r in log['retrievals'] if r['retrieved']]
    selected_fraud_amounts, candidate_fraud_amounts = [], []
    per_window_coverage = []
    overlaps = []
    for window in range(2, 7):
        records = [r for r in epochs if r['training_window'] == window]
        selected = [item['transaction_id'] for r in records for item in r['retrieved']]
        pool = materialize(log['memory_snapshots'][window-2])
        selected_fraud_amounts.extend(rows[i][3] for i in selected if rows[i][2] == 1)
        candidate_fraud_amounts.extend(r['amount'] for r in pool if r['class'] == 1)
        per_window_coverage.append(len(set(selected)) / len(pool))
        sets = [{item['transaction_id'] for item in r['retrieved']} for r in records]
        overlaps.extend(len(a & b) / len(a | b) for a, b in zip(sets, sets[1:]))
    return {'mean_historical_retrieval_microbenchmark_ms': float(np.mean(retrieval_benchmarks)*1000),
            'mean_window_reservoir_coverage_across_3_epochs': float(np.mean(per_window_coverage)),
            'mean_adjacent_epoch_jaccard': float(np.mean(overlaps)),
            'selected_fraud_mean_amount': float(np.mean(selected_fraud_amounts)),
            'candidate_fraud_mean_amount': float(np.mean(candidate_fraud_amounts)),
            'first_epoch_score_diagnostics': first_epoch_scores,
            'note': 'Scores reconstructed only at window starts from training-only checkpoints and reservoir rows. Not later-epoch score traces.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = {name: diagnose(Path(path)) for name, path in (entry.split('=', 1) for entry in args.run)}
    with args.output.open('x') as stream:
        stream.write(json.dumps(output, indent=2) + '\n')
    for name, row in output.items():
        print(name, {k: v for k, v in row.items() if k != 'first_epoch_score_diagnostics'})


if __name__ == '__main__':
    main()
