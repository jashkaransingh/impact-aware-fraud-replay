"""Synthetic regressions for the seed adapter; never reads real IEEE records."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ieee_multiseed as study
from ieee_multiseed import baseline, replay


class SeedAdapterTests(unittest.TestCase):
    def test_order_seed_is_independent_of_split(self):
        a = study.trace_orders(501, [256]*6)
        self.assertEqual(a, study.trace_orders(501, [256]*6))
        self.assertNotEqual(a, study.trace_orders(502, [256]*6))
        self.assertEqual(len(a), 18)
        self.assertEqual(study.SPLIT_SEED, 501)

    def test_legacy_501_equivalence_and_new_seed_on_synthetic_windows(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root/'windows'; data.mkdir()
            verification = {'holdout_rule': "synthetic fixed '501:<TransactionDT>'", 'preprocessing': {'feature_count': 3},
                'source_sha256_from_preparation': 'synthetic',
                'predictors': {'numeric': ['a', 'b', 'c'], 'categorical': [],
                               'excluded_from_model_inputs': ['TransactionAmt']},
                'verification': {k: True for k in ('development_rows_assigned_once',
                     'training_and_held_out_ids_disjoint', 'both_classes_in_each_subset')}}
            study.write(data/'verification.json', verification)
            study.write(data/'feature_names.json', ['a', 'b', 'c'])
            (data/'preprocessor.joblib').write_bytes(b'synthetic-only-preprocessor')
            rng = np.random.default_rng(8)
            for window in range(1, 7):
                for part, size in [('training', 256), ('held_out', 32)]:
                    offset = window*1000 + (0 if part == 'training' else 500)
                    frame = pd.DataFrame({'TransactionID': np.arange(offset, offset+size),
                        'TransactionDT': np.arange(offset, offset+size), 'isFraud': np.arange(size)%2,
                        'TransactionAmt': np.arange(size)+1.})
                    frame.to_csv(data/f'window-{window:02d}-{part}-metadata.csv', index=False)
                    sparse.save_npz(data/f'window-{window:02d}-{part}-features.npz',
                                    sparse.csr_matrix(rng.normal(size=(size, 3)).astype(np.float32)))
            manifest = study.preparation_manifest(data)
            sources = {name: study.sha(study.ROOT/name) for name in study.SOURCE_FILES}
            # Make new-run provenance use the real source hashes even though
            # legacy ROOT is scoped to the synthetic artifact directory.
            with patch.multiple(baseline, ROOT=root, INPUT=data), patch.multiple(replay, ROOT=root, INPUT=data, OUTPUT=replay.OUTPUT):
                original_base = root/'legacy-none'
                with patch.object(baseline, 'OUTPUT', original_base):
                    baseline.main()
                adapted_base = root/'adapter-none'
                study.run_method('no-replay', 501, adapted_base, manifest=manifest)
                self.assertEqual(study.read(original_base/'results.json'), study.read(adapted_base/'results.json'))
                for method in ('random', 'combined-classnorm'):
                    original = root/f'legacy-{method}'
                    options = ['--policy', 'combined' if method == 'combined-classnorm' else method,
                               '--output', str(original)]
                    if method == 'combined-classnorm':
                        options += ['--score-normalization', 'class']
                    with patch.object(replay, 'BASELINE', original_base), patch.object(sys, 'argv', ['synthetic', *options]):
                        replay.main()
                    adapted = root/f'adapter-{method}'
                    study.run_method(method, 501, adapted, adapted_base, manifest)
                    for name in ('results.json', 'replay-ids.json', 'smoke-check.json'):
                        self.assertEqual(study.read(original/name), study.read(adapted/name))
                    for window in range(1, 7):
                        name = f'checkpoint-window-{window:02d}.pt'
                        self.assertTrue(study.equal_state(torch.load(original/name, weights_only=True),
                                                         torch.load(adapted/name, weights_only=True)))
                second_base = root/'seed502-none'
                study.run_method('no-replay', 502, second_base, manifest=manifest)
                self.assertNotEqual(study.read(adapted_base/'results.json'), study.read(second_base/'results.json'))
                second_random = root/'seed502-random'
                study.run_method('random', 502, second_random, second_base, manifest)
                self.assertEqual(study.read(second_base/'results.json')[0]['held_out_evaluations'],
                                 study.read(second_random/'results.json')[0]['held_out_evaluations'])
                config = study.read(second_random/'configuration.json')
                self.assertEqual(config['training_seed'], 502)
                self.assertEqual(config['data_split_seed'], 501)
                self.assertIn('502', config['random_generators']['retrieval'])
                self.assertEqual(config['training_source_sha256'], sources)
                with self.assertRaises(FileExistsError):
                    study.run_method('no-replay', 502, second_base, manifest=manifest)
                with self.assertRaises(ValueError):
                    study.run_method('random', 503, root/'invalid', second_base, manifest)
            study.verify_hashes(manifest['files_sha256'])
            self.assertEqual(baseline.SEED, 501)
            self.assertEqual(replay.SEED, 501)

if __name__ == '__main__':
    unittest.main()
