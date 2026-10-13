import unittest
import numpy as np
from impact_replay.replay import Reservoir,score
from impact_replay.metrics import metrics,forgetting

class Invariants(unittest.TestCase):
 def test_exact_metrics(self):
  m=metrics([1,1,0],[.8,.1,.9],[10,90,20]);self.assertEqual(m['amount_recall'],.1);self.assertEqual(m['missed_fraud_amount'],90);self.assertEqual(m['false_positives'],1)
 def test_single_class(self):
  m=metrics([0,0],[.1,.2],[0,0]);self.assertIsNone(m['auc']);self.assertIsNone(m['ap']);self.assertIsNone(m['amount_recall'])
 def test_forgetting(self):
  self.assertAlmostEqual(forgetting([[.8,.6,.5],[np.nan,.7,.6],[np.nan,np.nan,.9]]),.2)
 def test_memory_and_policy(self):
  rows=[dict(id=i,y=i%2,reference_loss=.1,impact=i/100) for i in range(100)]
  a=Reservoir(16,1);b=Reservoir(16,1);a.add(rows);b.add(rows)
  self.assertEqual(a.rows,b.rows);self.assertEqual(len(a.rows),16);self.assertEqual(a.seen,100)
  for p in ['random','forgetting','impact','combined']:
   c=a.select(8,np.arange(16),p,np.random.default_rng(3));self.assertEqual(len(set(c)),8)
  z=Reservoir(0,1);z.add(rows);self.assertEqual(z.rows,[])
 def test_endpoints(self):
  f=np.array([0,.4,.2]);v=np.array([.9,0,.1])
  np.testing.assert_allclose(score(f,v,'combined',1),score(f,v,'forgetting'))
  np.testing.assert_allclose(score(f,v,'combined',0),score(f,v,'impact'))
 def test_no_replacement_small(self):
  a=Reservoir(3,1);a.add([dict(id=i,y=0,reference_loss=0,impact=0) for i in range(3)])
  self.assertEqual(len(a.select(10,np.zeros(3),'combined',np.random.default_rng(0))),3)
if __name__=='__main__':unittest.main()

class StratifiedTests(unittest.TestCase):
 def test_quota_and_capacity(self):
  from impact_replay.replay import StratifiedReservoir
  a=StratifiedReservoir(100,42,.5)
  a.add([dict(id=i,y=int(i%10==0),reference_loss=0,impact=i/1000) for i in range(1000)])
  self.assertEqual(len(a.rows),100);self.assertEqual(sum(r['y'] for r in a.rows),50)
  c=a.select(20,np.zeros(100),'impact',np.random.default_rng(1),fraud_fraction=.25)
  self.assertEqual(sum(a.rows[i]['y'] for i in c),5)
 def test_empty_and_bad_inputs(self):
  a=Reservoir(0,1);self.assertEqual(len(a.select(3,[],'random',np.random.default_rng(1))),0)
  with self.assertRaises(ValueError):score([1],[1],'made_up')

class StorageTests(unittest.TestCase):
 def test_atomic_artifacts(self):
  import tempfile,torch
  from pathlib import Path
  from impact_replay.storage import save_npz,save_checkpoint
  with tempfile.TemporaryDirectory() as td:
   a=Path(td)/'a.npz';save_npz(a,x=np.arange(10000));np.testing.assert_array_equal(np.load(a)['x'],np.arange(10000))
   b=Path(td)/'b.pt';save_checkpoint({'x':torch.arange(100)},b);self.assertTrue(torch.equal(torch.load(b)['x'],torch.arange(100)))
