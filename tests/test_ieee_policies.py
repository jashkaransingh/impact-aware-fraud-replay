"""Small synthetic checks; never loads IEEE data or writes result artifacts."""
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_ieee_balanced_replay as runner


class IEEEPolicies(unittest.TestCase):
    def memory(self, policy):
        memory = runner.StratifiedReservoir(random.Random(501), random.Random(501))
        memory.policy = policy
        memory.impact_scale = 1.
        return memory

    def test_policies_and_guards(self):
        matrix = sparse.csr_matrix(np.zeros((140, 2), dtype=np.float32))
        metadata = pd.DataFrame({'TransactionID': range(140),
                                 'isFraud': [0, 1] * 70,
                                 'TransactionAmt': np.arange(140)})
        model = runner.Network(2)
        snapshots = []
        for policy in ('random', 'forgetting', 'impact', 'combined'):
            memory = self.memory(policy)
            memory.admit_window(matrix, metadata, 1, {999}, model=model)
            snapshots.append(memory.snapshot(1))
            rejected = self.memory(policy)
            with self.assertRaises(AssertionError):
                rejected.admit_window(matrix, metadata, 1, {0}, model=model)
            references = [r['reference_loss'] for p in memory.slots.values() for r in p]
            chosen = memory.retrieve(2, {1: set(range(140))}, {999}, model=model)
            self.assertEqual(len(chosen), 128)
            self.assertEqual(sum(r['class'] for r in chosen), 64)
            self.assertEqual(len({r['transaction_id'] for r in chosen}), 128)
            self.assertEqual(references, [r['reference_loss'] for p in memory.slots.values() for r in p])
            with self.assertRaises(AssertionError):
                memory.retrieve(1, {1: set(range(140))}, {999}, model=model)
            with self.assertRaises(AssertionError):
                memory.retrieve(2, {1: set(range(140))}, {0}, model=model)
        self.assertTrue(all(s == snapshots[0] for s in snapshots))

    def test_rankings_and_training_step(self):
        from impact_replay.replay import score
        model = runner.Network(2)
        for policy in ('forgetting', 'impact', 'combined'):
            memory = self.memory(policy)
            memory.slots = {c: [dict(transaction_id=2*i+c, source_window=1,
                                   features=np.zeros(2, dtype=np.float32),
                                   reference_loss=i/100, impact=(i/80)*c,
                                   **{'class': c}) for i in range(80)] for c in (0, 1)}
            rows = sum(memory.slots.values(), [])
            losses = runner.historical_losses(model, rows)
            scores = score(losses - [r['reference_loss'] for r in rows],
                           [r['impact'] for r in rows], policy)
            selected = memory.retrieve(2, {1: set(range(160))}, set(), model=model)
            chosen_ids = {r['transaction_id'] for r in selected}
            for c in (0, 1):
                kept = [v for r, v in zip(rows, scores) if r['class'] == c and r['transaction_id'] in chosen_ids]
                dropped = [v for r, v in zip(rows, scores) if r['class'] == c and r['transaction_id'] not in chosen_ids]
                self.assertGreaterEqual(min(kept), max(dropped))
            x = torch.from_numpy(np.stack([r['features'] for r in selected]))
            y = torch.tensor([r['class'] for r in selected], dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=runner.LEARNING_RATE)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            self.assertTrue(model.training)

    def test_candidate_normalization_and_endpoints(self):
        rows = [dict(reference_loss=0., impact=v, **{'class': c})
                for c, v in [(0, 0), (0, 0), (1, .1), (1, .9)]]
        losses = np.array([100., 200., .1, .2])
        global_scores = runner.candidate_scores(rows, losses, 'combined')
        class_scores = runner.candidate_scores(rows, losses, 'combined', normalization='class')
        self.assertGreater(class_scores[2], global_scores[2])
        self.assertGreater(class_scores[3], global_scores[3])
        np.testing.assert_allclose(
            runner.candidate_scores(rows, losses, 'combined', alpha=1),
            runner.candidate_scores(rows, losses, 'forgetting'))
        np.testing.assert_allclose(
            runner.candidate_scores(rows, losses, 'combined', alpha=0),
            runner.candidate_scores(rows, losses, 'impact'))
        with self.assertRaises(ValueError):
            runner.candidate_scores(rows, losses, 'combined', normalization='invalid')

    def test_mixture_exact_quota_and_no_duplicate_ids(self):
        memory = self.memory('combined')
        memory.random_fraction = .5
        memory.slots = {c: [dict(transaction_id=2*i+c, source_window=1,
                               features=np.zeros(2, dtype=np.float32),
                               reference_loss=0., impact=i/80*c, **{'class': c})
                           for i in range(80)] for c in (0, 1)}
        model = runner.Network(2)
        expected = []
        rng = random.Random(501)
        rows = sum(memory.slots.values(), [])
        values = runner.candidate_scores(rows, runner.historical_losses(model, rows), 'combined')
        values_by_id = {r['transaction_id']: v for r, v in zip(rows, values)}
        for c in (0, 1):
            pool = memory.slots[c]
            ranking = rng.sample(pool, len(pool))
            ranking.sort(key=lambda r: -values_by_id[r['transaction_id']])
            uniform = rng.sample(pool, 32)
            ids = {r['transaction_id'] for r in uniform}
            priority = [r for r in ranking if r['transaction_id'] not in ids][:32]
            expected.extend(uniform + priority)
        rng.shuffle(expected)
        actual = memory.retrieve(2, {1: set(range(160))}, set(), model=model)
        self.assertEqual([r['transaction_id'] for r in actual], [r['transaction_id'] for r in expected])
        self.assertEqual(len({r['transaction_id'] for r in actual}), 128)
        self.assertEqual(sum(r['class'] for r in actual), 64)

    def test_random_compatibility(self):
        memory = self.memory('random')
        memory.slots = {c: [dict(transaction_id=2*i+c, source_window=1, **{'class': c})
                            for i in range(80)] for c in (0, 1)}
        rng = random.Random(501)
        expected = sum((rng.sample(memory.slots[c], 64) for c in (0, 1)), [])
        rng.shuffle(expected)
        self.assertEqual(memory.retrieve(2, {1: set(range(160))}, set()), expected)

    def test_forgetting_ignores_amount(self):
        memory = self.memory('forgetting')
        memory.slots = {c: [dict(transaction_id=2*i+c, source_window=1,
                               features=np.zeros(2, dtype=np.float32),
                               reference_loss=i/100, impact=0., **{'class': c})
                           for i in range(80)] for c in (0, 1)}
        model = runner.Network(2)
        first = memory.retrieve(2, {1: set(range(160))}, set(), model=model)
        for pool in memory.slots.values():
            for item in pool:
                item['impact'] = 1000 - item['transaction_id']
        memory.retrieval_rng = random.Random(501)
        second = memory.retrieve(2, {1: set(range(160))}, set(), model=model)
        self.assertEqual([r['transaction_id'] for r in first], [r['transaction_id'] for r in second])

if __name__ == '__main__':
    unittest.main()
