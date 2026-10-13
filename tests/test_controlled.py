import unittest
import numpy as np
from impact_replay.controlled import make_block
class ControlledTests(unittest.TestCase):
 def test_amount_not_label_or_feature(self):
  c={'fraud_fraction_data':.05,'feature_center':2.,'feature_sd':.8,'amount_mode':'independent'}
  a=make_block(2000,0,np.random.default_rng(99),c,0);b=make_block(2000,0,np.random.default_rng(99),dict(c,amount_mode='a_high'),0)
  np.testing.assert_array_equal(a['x'],b['x']);np.testing.assert_array_equal(a['y'],b['y']);self.assertEqual(int(a['y'].sum()),100)
  np.testing.assert_allclose(b['amount'][a['y']==1],10*a['amount'][a['y']==1]);np.testing.assert_array_equal(b['amount'][a['y']==0],a['amount'][a['y']==0])
 def test_latent_rule_consistent(self):
  c={'fraud_fraction_data':.05,'feature_center':2.,'feature_sd':0.,'amount_mode':'independent'}
  for d in [0,1]:
   b=make_block(100,d,np.random.default_rng(1),c,d*100)
   np.testing.assert_array_equal((b['x'][:,0]*b['x'][:,1]>0).astype(int),b['y'])
if __name__=='__main__':unittest.main()
