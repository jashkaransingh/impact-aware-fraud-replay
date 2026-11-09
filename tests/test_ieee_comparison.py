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
        self.assertEqual(results, before)
        with self.assertRaises(ValueError):
            summarize(results[:5])
