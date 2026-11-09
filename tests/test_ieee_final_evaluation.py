"""Synthetic-only final protocol integration; no real raw partition is opened."""
import copy
import csv
from importlib.metadata import version
import io
import statistics
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ieee_final_evaluation as final
from ieee_final_data import partition_at, load_partition, transform_frame, validate_preparation
from ieee_multiseed import ROOT, SEEDS, METHODS, SOURCE_FILES, sha, write, read
from prepare_ieee_windows import PREDICTORS, CATEGORICAL, NUMERIC, METADATA, build_preprocessor
from run_ieee_no_replay_baseline import Network, evaluate


class SyntheticFinalFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        times = [0, 20, 40, 59, 60, 70, 79, 80, 90, 100]
        rows = []
        for i, dt in enumerate(times):
            row = {name: ('A' if i == 0 else 'B' if i == 1 else 'FUTURE') for name in CATEGORICAL}
            row.update(dist1=[1., 3., 400., 500., 900., 1000., 1100., 1200., 1300., 1400.][i], dist2=i+1.,
                       TransactionID=100+i, TransactionDT=dt, TransactionAmt=10.+i, isFraud=i%2)
            rows.append(row)
        self.frame = pd.DataFrame(rows)
        self.source = self.root/'synthetic.csv'
        self.frame.to_csv(self.source, index=False)
        self.prepared = {'source': str(self.source), 'source_bytes': self.source.stat().st_size,
            'source_sha256': sha(self.source), 'source_row_count': 10,
            'time_span': {'min_transaction_dt': 0, 'max_transaction_dt': 100},
            'partition_checks': {'partitions_chronological': True, 'partitions_nonoverlapping': True},
            'partitions': {'development': {'row_count': 4, 'min_transaction_dt': 0, 'max_transaction_dt': 59},
                           'validation': {'row_count': 3, 'min_transaction_dt': 60, 'max_transaction_dt': 79},
                           'reserved_evaluation': {'row_count': 3, 'min_transaction_dt': 80, 'max_transaction_dt': 100}}}
        self.preparation = self.root/'preparation.json'; write(self.preparation, self.prepared)
        self.windows = self.root/'windows'; self.windows.mkdir()
        self.processor = build_preprocessor().fit(self.frame.iloc[:2][PREDICTORS])
        self.names = self.processor.get_feature_names_out().tolist()
        joblib.dump(self.processor, self.windows/'preprocessor.joblib')
        write(self.windows/'feature_names.json', self.names)
        write(self.windows/'verification.json', {'source_sha256_from_preparation': self.prepared['source_sha256'],
            'preprocessing': {'fit_scope': 'window 1 training subset only', 'frozen_for_later_windows': True,
                              'feature_count': len(self.names)},
            'predictors': {'excluded_from_model_inputs': METADATA}})
        for w in range(1, 7):
            for part in ('training', 'held_out'):
                self.frame.iloc[:2][METADATA].to_csv(self.windows/f'window-{w:02d}-{part}-metadata.csv', index=False)
                sparse.save_npz(self.windows/f'window-{w:02d}-{part}-features.npz', sparse.csr_matrix(np.zeros((2,len(self.names)), dtype=np.float32)))
        self.data_hashes = {str(p): sha(p) for p in self.windows.iterdir()}
        versions = {name: version(name) for name in ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib')}
        records = []
        for seed in SEEDS:
            for method in METHODS:
                path = self.root/f'{seed}-{method}'; path.mkdir()
                torch.manual_seed(seed)
                model = Network(len(self.names))
                torch.save({'after_training_window': 6, 'seed': seed, 'input_width': len(self.names),
                            'model_state': model.state_dict()}, path/'checkpoint-window-06.pt')
                config = {'seed': seed, 'training_seed': seed, 'data_split_seed': 501,
                    'input_width': len(self.names), 'device': 'cpu', 'epochs_per_window': 3,
                    'learning_rate': .001, 'batch_size': 256, 'positive_weight': 1., 'versions': versions,
                    'source_preparation_sha256': self.prepared['source_sha256'], 'prepared_files_sha256': self.data_hashes}
                if method != 'no-replay':
                    config.update(policy='combined' if method == 'combined-classnorm' else method,
                                  score_normalization='class' if method == 'combined-classnorm' else 'global')
                write(path/'configuration.json', config)
                records.append({'seed': seed, 'method': method, 'path': str(path), 'status': 'reused' if seed==501 else 'passed'})
        self.registry = self.root/'registry.json'
        write(self.registry, {'status': 'complete', 'seeds': list(SEEDS), 'methods': list(METHODS), 'runs': records,
             'prepared_data': {'files_sha256': self.data_hashes},
             'source_sha256': {name: sha(ROOT/name) for name in SOURCE_FILES}})
        self.contract = final.build_contract(self.preparation, self.windows, self.registry)

    def test_contract_is_metadata_only_and_rejects_nontraining_fit(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.resolve() == self.source.resolve():
                raise AssertionError('Raw source opened during metadata check')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            contract = final.build_contract(self.preparation, self.windows, self.registry)
            final.verify_contract(contract)
        self.assertEqual(len(contract['models']), 25)
        verification = read(self.windows/'verification.json')
        verification['preprocessing']['fit_scope'] = 'validation'
        write(self.windows/'verification.json', verification)
        with self.assertRaisesRegex(ValueError, 'Training-only'):
            final.build_contract(self.preparation, self.windows, self.registry)

    def test_frozen_policy_and_training_budget_changes_are_rejected(self):
        path = Path(next(r for r in self.contract['models'] if r['method'] == 'combined-classnorm')['configuration'])
        original = read(path)
        for key, value in [('alpha', .75), ('batch_size', 128)]:
            changed = dict(original); changed[key] = value; write(path, changed)
            with self.assertRaises(ValueError):
                final.build_contract(self.preparation, self.windows, self.registry)
        write(path, original)
        final.verify_contract(self.contract)

    def test_validation_reader_never_requests_reserved_row(self):
        original = Path.open
        requests = []
        class PrefixOnly:
            def __enter__(inner):
                inner.stream = original(self.source, 'rb'); return inner
            def __exit__(inner, *args):
                inner.stream.close()
            def readline(inner):
                if len(requests) >= 8:  # header + 4 development + 3 validation
                    raise AssertionError('Reserved row requested by validation')
                requests.append(1)
                return inner.stream.readline()
        def guarded(path, *args, **kwargs):
            return PrefixOnly() if path.resolve() == self.source.resolve() else original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            frame, lineage = load_partition(self.source, self.prepared, 'validation')
        self.assertEqual(len(requests), 8)
        self.assertEqual(frame.TransactionDT.tolist(), [60,70,79])
        self.assertFalse(lineage['full_source_verified'])

    def test_transform_only_and_outcomes_cannot_change_predictions(self):
        self.assertAlmostEqual(self.processor.named_transformers_['numeric'].named_steps['scaler'].mean_[0], 2.)
        later = self.frame.iloc[4:].copy()
        changed = later.copy()
        changed['TransactionAmt'] = 1e12
        changed['isFraud'] = 1-changed['isFraud']
        changed['TransactionID'] = changed.TransactionID+999
        with patch.object(ColumnTransformer, 'fit', side_effect=AssertionError('Refit')):
            a = transform_frame(later, self.processor, self.names)
            b = transform_frame(changed, self.processor, self.names)
        np.testing.assert_array_equal(a.toarray(), b.toarray())
        model = Network(len(self.names)); before = copy.deepcopy(model.state_dict())
        np.testing.assert_array_equal(final.predict(model, a), final.predict(model, b))
        self.assertTrue(all(torch.equal(before[k], model.state_dict()[k]) for k in before))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all('FUTURE' not in str(cats) for cats in self.processor.named_transformers_['categorical'].categories_))

    def test_full_synthetic_pipeline_predictions_precede_labels_no_training_or_replay(self):
        val_output = self.root/'validation-results'
        original_score = final.score_predictions
        def score_after_persistence(scores, frame):
            self.assertEqual(len(list(val_output.glob('predictions-*.npz'))), 25)
            self.assertTrue((val_output/'predictions-complete.json').exists())
            return original_score(scores, frame)
        before_hashes = {r['checkpoint']: sha(r['checkpoint']) for r in self.contract['models']}
        with patch.object(ColumnTransformer, 'fit', side_effect=AssertionError('Refit')), \
             patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Training')), \
             patch('run_ieee_balanced_replay.StratifiedReservoir.retrieve', side_effect=AssertionError('Replay')), \
             patch('run_ieee_balanced_replay.StratifiedReservoir.admit_window', side_effect=AssertionError('Admission')), \
             patch.object(final, 'score_predictions', side_effect=score_after_persistence):
            metrics = final.execute_phase(self.contract, 'validation', val_output)
        self.assertEqual(len(metrics['per_seed']), 25)
        self.assertEqual(self.contract['settings']['memory_per_class'], 128)
        self.assertEqual(self.contract['settings']['retrieve_per_class'], 64)
        self.assertEqual(self.contract['settings']['review_fraction'], .05)
        reserved = final.execute_phase(self.contract, 'reserved', self.root/'reserved-results', val_output/'receipt.json')
        self.assertTrue(read(self.root/'reserved-results'/'receipt.json')['data']['full_source_verified'])
        self.assertEqual(len(reserved['paired_differences']), 5)
        self.assertEqual(before_hashes, {r['checkpoint']: sha(r['checkpoint']) for r in self.contract['models']})
        with self.assertRaises(FileExistsError):
            final.execute_phase(self.contract, 'validation', val_output)

    def test_reserved_gates_fail_before_source_access(self):
        with patch.object(final, 'load_partition', side_effect=AssertionError('Source opened')) as loader:
            with self.assertRaisesRegex(ValueError, 'validation receipt'):
                final.execute_phase(self.contract, 'reserved', self.root/'blocked')
            loader.assert_not_called()
        self.assertFalse((self.root/'blocked').exists())
        bad = copy.deepcopy(self.contract); bad['settings']['alpha'] = .75
        with self.assertRaisesRegex(ValueError, 'digest'):
            final.verify_contract(bad)
        checkpoint = Path(self.contract['models'][0]['checkpoint'])
        checkpoint.write_bytes(b'changed')
        with patch.object(final, 'load_partition', side_effect=AssertionError('Source opened')) as loader:
            with self.assertRaises(AssertionError):
                final.execute_phase(self.contract, 'validation', self.root/'bad-checkpoint')
            loader.assert_not_called()

    def test_changed_validation_prediction_receipt_blocks_reserved(self):
        output = self.root/'validation'
        final.execute_phase(self.contract, 'validation', output)
        prediction = next(output.glob('predictions-*.npz'))
        prediction.write_bytes(b'changed')
        with patch.object(final, 'load_partition', side_effect=AssertionError('Source opened')) as loader:
            with self.assertRaises(AssertionError):
                final.execute_phase(self.contract, 'reserved', self.root/'blocked', output/'receipt.json')
            loader.assert_not_called()

    def test_chronology_boundaries_duplicate_ids_and_source_integrity(self):
        self.assertEqual(partition_at(60,0,100), 'validation')
        self.assertEqual(partition_at(80,0,100), 'reserved_evaluation')
        bad = copy.deepcopy(self.prepared)
        bad['partitions']['validation']['min_transaction_dt'] = 59
        with self.assertRaises(ValueError):
            validate_preparation(bad)
        changed = self.frame.copy(); changed.loc[4,'TransactionID'] = changed.loc[0,'TransactionID']
        changed.to_csv(self.source,index=False)
        prep = copy.deepcopy(self.prepared); prep['source_bytes'] = self.source.stat().st_size
        with self.assertRaisesRegex(ValueError, 'uniqueness'):
            load_partition(self.source, prep, 'validation')
        self.frame.to_csv(self.source,index=False)
        _, lineage = load_partition(self.source, self.prepared, 'validation')
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            load_partition(self.source, self.prepared, 'reserved', expected_prefix='0'*64)
        prep = copy.deepcopy(self.prepared); prep['source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            load_partition(self.source, prep, 'reserved', lineage['validation_prefix_sha256'])


class FinalMetricTests(unittest.TestCase):
    def test_exact_metrics_quota_ties_and_legacy_equivalence(self):
        labels = np.zeros(20,dtype=int); labels[[0,19]] = 1
        amounts = np.ones(20); amounts[[0,19]] = [10.,90.]
        meta = pd.DataFrame({'TransactionID': np.arange(20), 'isFraud':labels,'TransactionAmt':amounts})
        scores = np.linspace(.95,.05,20)
        result = final.score_predictions(scores, meta)
        self.assertAlmostEqual(result['ap'], .55)
        self.assertEqual(result['review_quota_rows'],1)
        self.assertEqual(result['fraud_recall_at_5pct'],.5)
        self.assertEqual(result['amount_weighted_fraud_recall_at_5pct'],.1)
        meta['TransactionID'] = np.arange(20)[::-1]
        tied = final.score_predictions(np.ones(20)*.5,meta)
        self.assertEqual(tied['amount_weighted_fraud_recall_at_5pct'],.9)
        class LogitModel(torch.nn.Module):
            def forward(self,x): return x[:,0]
        matrix = sparse.csr_matrix(np.log(scores/(1-scores)).reshape(-1,1).astype(np.float32))
        model = LogitModel()
        legacy = evaluate(model,matrix,meta)
        separated = final.score_predictions(final.predict(model,matrix),meta)
        for key in separated:
            self.assertEqual(legacy[key],separated[key])
        larger = pd.concat([meta,meta.iloc[:1].assign(TransactionID=999)],ignore_index=True)
        self.assertEqual(final.score_predictions(np.ones(21)*.5,larger)['review_quota_rows'],2)

    def test_primary_aggregation_pairs_by_seed_and_uses_sample_sd(self):
        rows = [{'seed': seed, 'method': method,
                 **{metric: .2+(seed-501)*.01+((seed-500)*.001 if method == 'combined-classnorm' else 0.)
                    for metric in final.METRICS}} for seed in SEEDS for method in METHODS]
        report = final.aggregate_metrics(list(reversed(rows)))
        primary = report['contrasts']['amount_weighted_fraud_recall_at_5pct']
        self.assertAlmostEqual(primary['mean'], .003)
        self.assertAlmostEqual(primary['sd'], statistics.stdev([.001,.002,.003,.004,.005]))
        self.assertEqual(primary['positive'], 5)
        self.assertEqual([r['seed'] for r in report['paired_differences']], list(SEEDS))
        self.assertLess(primary['paired_t_95_interval'][0], .003)
        self.assertGreater(primary['paired_t_95_interval'][1], .003)

    def test_undefined_and_invalid_metrics(self):
        meta = pd.DataFrame({'TransactionID':[1,2], 'isFraud':[0,0], 'TransactionAmt':[0.,0.]})
        result = final.score_predictions([.1,.2],meta)
        self.assertIsNone(result['ap']); self.assertIsNone(result['amount_weighted_fraud_recall_at_5pct'])
        meta['isFraud'] = [0,1]
        self.assertIsNone(final.score_predictions([.1,.2],meta)['amount_weighted_fraud_recall_at_5pct'])
        with self.assertRaises(ValueError): final.score_predictions([np.nan,.2],meta)
        meta['TransactionID'] = [1,1]
        with self.assertRaises(ValueError): final.score_predictions([.1,.2],meta)
        rows = [{'seed':s,'method':m,**{metric:None for metric in final.METRICS}} for s in SEEDS for m in METHODS]
        report = final.aggregate_metrics(rows)
        self.assertFalse(report['contrasts']['amount_weighted_fraud_recall_at_5pct']['available'])
        with self.assertRaises(ValueError): final.aggregate_metrics(rows[:-1])

if __name__ == '__main__': unittest.main()
