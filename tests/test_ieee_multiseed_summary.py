import copy
from pathlib import Path
import sys
import unittest
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_ieee_multiseed import aggregate
from ieee_multiseed import SEEDS, METHODS, METRICS


class PairedSummaryTests(unittest.TestCase):
    def rows(self):
        return [{'seed': seed, 'method': method, **{metric: .2+(seed-501)*.01+
                 ((seed-500)*.001 if method == 'combined-classnorm' else 0.) for metric in METRICS}}
                for seed in SEEDS for method in METHODS]

    def test_sample_sd_pairing_and_difference_direction(self):
        rows = self.rows()
        before = copy.deepcopy(rows)
        # Order must not influence pairing: pairing uses seed and method keys.
        report = aggregate(list(reversed(rows)))
        original = next(r for r in report['summary'] if r['method'] == 'combined')
        self.assertAlmostEqual(original['historical_ap']['mean'], .22)
        self.assertAlmostEqual(original['historical_ap']['sd'], statistics.stdev([.2, .21, .22, .23, .24]))
        self.assertEqual([r['seed'] for r in report['paired_differences']], list(SEEDS))
        for index, row in enumerate(report['paired_differences'], 1):
            self.assertAlmostEqual(row['historical_ap'], index*.001)
        primary = report['paired_summary']['historical_amount_recall_5pct']
        self.assertAlmostEqual(primary['mean'], .003)
        self.assertAlmostEqual(primary['sd'], statistics.stdev([.001, .002, .003, .004, .005]))
        self.assertAlmostEqual(primary['new_seeds_502_to_505_mean'], .0035)
        self.assertEqual(primary['improvements'], 5)
        self.assertEqual(report['paired_summary']['forgetting']['improvements'], 0)
        lo, hi = primary['paired_t_95_interval']
        self.assertLess(lo, .003); self.assertGreater(hi, .003)
        self.assertEqual(rows, before)

    def test_reject_missing_duplicate_seed_or_nonfinite(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        duplicate = copy.deepcopy(rows)
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            aggregate(duplicate)
        rows[0]['historical_ap'] = float('nan')
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_zero_difference_interval_and_ties(self):
        rows = self.rows()
        for row in rows:
            for metric in METRICS:
                row[metric] = .2
        primary = aggregate(rows)['paired_summary']['historical_amount_recall_5pct']
        self.assertEqual(primary['paired_t_95_interval'], [0., 0.])
        self.assertEqual(primary['ties'], 5)
