"""Synthetic metadata checks for the read-only reserved audit; no predictions run."""
from pathlib import Path
import sys
import tempfile
import unittest
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ieee_multiseed import SEEDS,METHODS,sha,write
from report_ieee_reserved import verify_reserved_receipt


class ReservedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.contract={'contract_sha256':'c'*64,'preparation':{'source_row_count':10,'source_sha256':'a'*64,
                       'partitions':{'reserved_evaluation':{'row_count':3}}}}
        files={}
        for seed in SEEDS:
            for method in METHODS:
                name=f'predictions-{seed}-{method}.npz'
                (self.path/name).write_bytes(b'synthetic hash fixture only')
                files[name]=sha(self.path/name)
        write(self.path/'predictions-complete.json',{'contract_sha256':'c'*64,'files_sha256':files})
        write(self.path/'metrics.json',{'synthetic':True});files['metrics.json']=sha(self.path/'metrics.json')
        self.receipt={'status':'complete','phase':'reserved','contract_sha256':'c'*64,
            'validation_receipt_sha256':'b'*64,'updates_after_development':0,
            'all_predictions_saved_before_metrics':True,'files_sha256':files,
            'data':{'phase':'reserved','rows':3,'consumed_rows':10,'full_source_verified':True,
                    'consumed_source_sha256':'a'*64}}
        write(self.path/'receipt.json',self.receipt)
        write(self.path/'status.json',{'status':'complete'})

    def test_requires_full_source_and_unchanged_validation_lineage(self):
        self.assertEqual(verify_reserved_receipt(self.path,self.contract,'b'*64)['data']['rows'],3)
        for field,value in [('full_source_verified',False),('consumed_rows',9),('consumed_source_sha256','0'*64)]:
            bad=copy.deepcopy(self.receipt);bad['data'][field]=value;write(self.path/'receipt.json',bad)
            with self.assertRaises(ValueError):verify_reserved_receipt(self.path,self.contract,'b'*64)
        write(self.path/'receipt.json',self.receipt)
        with self.assertRaises(ValueError):verify_reserved_receipt(self.path,self.contract,'0'*64)

    def test_tampered_or_missing_prediction_set_is_rejected(self):
        bad=copy.deepcopy(self.receipt);bad['files_sha256'].pop(f'predictions-{SEEDS[0]}-{METHODS[0]}.npz')
        write(self.path/'receipt.json',bad)
        with self.assertRaises(ValueError):verify_reserved_receipt(self.path,self.contract,'b'*64)
        write(self.path/'receipt.json',self.receipt)
        (self.path/f'predictions-{SEEDS[0]}-{METHODS[0]}.npz').write_bytes(b'changed')
        with self.assertRaises(AssertionError):verify_reserved_receipt(self.path,self.contract,'b'*64)
