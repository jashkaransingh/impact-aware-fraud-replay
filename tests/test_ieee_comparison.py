import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_ieee_policies import summarize


class ComparisonTests(unittest.TestCase):
    def test_known_trajectories(self):
        results = []
        for step in range(1, 7):
            results.append({'after_training_window': step, 'held_out_evaluations': [
                {'held_out_window': window, 'ap': 1 - (step-window)*.1,
                 'amount_weighted_fraud_recall_at_5pct': .5, 'fraud_recall_at_5pct': .25}
                for window in range(1, step+1)]})
        before = copy.deepcopy(results)
        row = summarize(results)
        self.assertAlmostEqual(row['historical_ap'], .7)
        self.assertAlmostEqual(row['forgetting'], .3)
        self.assertEqual(row['current_window_ap'], 1)
        self.assertEqual(row['historical_amount_recall_5pct'], .5)
        self.assertEqual(row['mean_diagonal_ap'], 1)
        from compare_ieee_policies import per_window
        window = per_window(results)[0]
        self.assertEqual(window['initial_ap'], 1)
        self.assertEqual(window['final_ap'], .5)
        self.assertEqual(window['forgetting'], .5)
        self.assertEqual(window['final_amount_recall_5pct'], .5)
        self.assertEqual(results, before)
        with self.assertRaises(ValueError):
            summarize(results[:5])

    def test_audit_rejects_id_outside_prior_memory(self):
        from unittest.mock import patch
        import compare_ieee_policies as comparison
        snapshots = [{'slots_by_class': {'0': [{'transaction_id': 0, 'source_window': 1}],
                                         '1': [{'transaction_id': 1, 'source_window': 1}]}}
                     for _ in range(6)]
        entries = [{'training_window': w, 'epoch': e, 'current_rows_exposed': 10,
                    'optimizer_steps': 1, 'retrieved': ([] if w == 1 else [
                        {'transaction_id': c, 'class': c, 'source_window': 1} for c in (0, 1)])}
                   for w in range(1, 7) for e in range(1, 4)]
        anchor = {'memory_snapshots': snapshots, 'retrievals': entries}
        candidate = copy.deepcopy(anchor)
        results = [{'training_rows': 10, 'held_out_evaluations': [], 'epoch_mean_losses': [1, 1, 1]} for _ in range(6)]
        config = {k: 1 for k in ('seed', 'input_width', 'positive_weight', 'batch_size_current_examples',
                                  'epochs_per_window', 'learning_rate', 'source_sha256_from_preparation', 'versions')}
        def fake_read(path):
            return {'configuration.json': config, 'replay-ids.json': candidate,
                    'results.json': results, 'verification.json': {'ok': True}}[path.name]
        with patch.object(comparison, 'read', side_effect=fake_read):
            comparison.audit_run(Path('/synthetic'), results, anchor)
            candidate['retrievals'][3]['retrieved'][0]['transaction_id'] = 999
            with self.assertRaisesRegex(AssertionError, 'outside prior reservoir'):
                comparison.audit_run(Path('/synthetic'), results, anchor)
