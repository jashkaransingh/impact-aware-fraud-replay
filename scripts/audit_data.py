import hashlib,json,pickle
from pathlib import Path
import pandas as pd
import numpy as np
root=Path(__file__).resolve().parents[1];base=root/'upstream/POCL/dataset'
d=pd.read_csv(base/'claims.csv');b=pd.read_csv(base/'bene.csv')
a={'claims':len(d),'beneficiaries':len(b),'providers':int(d.Provider.nunique()),'fraud_rate':float(d.PotentialFraud.mean()),'providers_with_mixed_labels':int((d.groupby('Provider').PotentialFraud.nunique()>1).sum()),'amount_summary':d.InscClaimAmtReimbursed.describe().to_dict(),'dates':[d.ClaimStartDt.min(),d.ClaimStartDt.max()],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob('*.csv')}}
for name in ['datasetonline','datasettwo']:
 with open(root/f'work/pocl/Dataset/{name}.dat','rb') as f:ds=pickle.load(f)
 a[name]={'graphs':len(ds),'nodes':sum(len(x) for x,e in ds),'features_including_label':ds[0][0].shape[1],'finite':all(np.isfinite(x).all() for x,e in ds),'edge_valid':all(e.shape[0]==2 and (not e.size or (e.min()>=0 and e.max()<len(x))) for x,e in ds)}
(root/'results/data-audit.json').write_text(json.dumps(a,indent=2))
