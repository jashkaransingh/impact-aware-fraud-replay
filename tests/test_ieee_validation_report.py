"""Small checks for independent validation reporting; existing tests unchanged."""
from pathlib import Path
import sys
import unittest
import statistics
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_ieee_validation import independent_metrics,summarize_rows
from ieee_final_evaluation import METRICS
from ieee_multiseed import METHODS,SEEDS


class ValidationReportTests(unittest.TestCase):
    def test_independent_quota_amount_denominator_and_id_ties(self):
        labels=np.zeros(20,dtype=int);labels[[0,19]]=1
        amounts=np.ones(20);amounts[[0,19]]=[10,90]
        frame=pd.DataFrame({'TransactionID':np.arange(20),'isFraud':labels,'TransactionAmt':amounts})
        result=independent_metrics(np.linspace(.95,.05,20),frame)
        self.assertAlmostEqual(result['ap'],.55)
        self.assertEqual(result['review_quota_rows'],1)
        self.assertEqual(result['fraud_recall_at_5pct'],.5)
        self.assertEqual(result['amount_weighted_fraud_recall_at_5pct'],.1)
        frame['TransactionID']=np.arange(20)[::-1]
        self.assertEqual(independent_metrics(np.ones(20)*.5,frame)['amount_weighted_fraud_recall_at_5pct'],.9)

    def test_paired_seed_lookup_and_sample_sd(self):
        rows=[{'seed':s,'method':m,**{metric:.2+(s-501)*.01+((s-500)*.001 if m=='combined-classnorm' else 0)
              for metric in METRICS}} for s in SEEDS for m in METHODS]
        summary,paired,contrasts=summarize_rows(list(reversed(rows)))
        self.assertAlmostEqual(contrasts[METRICS[-1]]['mean'],.003)
        self.assertAlmostEqual(contrasts[METRICS[-1]]['sd'],statistics.stdev([.001,.002,.003,.004,.005]))
        self.assertEqual([r['seed'] for r in paired],list(SEEDS))
        with self.assertRaises(ValueError):summarize_rows(rows[:-1])
