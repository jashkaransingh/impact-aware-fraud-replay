import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'upstream/handbook'))
import simulator
p=argparse.ArgumentParser();p.add_argument('--days',type=int,default=84);a=p.parse_args()
out=root/'data';out.mkdir(exist_ok=True)
c,t,d=simulator.generate_dataset(n_customers=1000,n_terminals=10000,nb_days=a.days,r=10)
d=simulator.add_frauds(c,t,d);d=d.sort_values(['TX_DATETIME','TRANSACTION_ID']).reset_index(drop=True)
d.to_csv(out/'handbook.csv',index=False)
meta={'generator_commit':'81cf7d1714bb7b2f5b496407d9055d91dc68dc25','customers':1000,'terminals':10000,'days':a.days,'radius':10,'rows':len(d),'frauds':int(d.TX_FRAUD.sum()),'scenarios':{str(k):int(v) for k,v in d.TX_FRAUD_SCENARIO.value_counts().items()},'sha256':hashlib.sha256((out/'handbook.csv').read_bytes()).hexdigest()}
(out/'provenance.json').write_text(json.dumps(meta,indent=2));print(meta)
