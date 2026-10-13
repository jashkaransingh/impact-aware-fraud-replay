import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from impact_replay.metrics import metrics
r=Path('results');report={}
for prefix in ['calibrated','final']:
 a=r/f'{prefix}-combined-17';b=r/f'{prefix}-combined-17-repeat'
 report[prefix+'_repeat']={name:(a/name).read_bytes()==(b/name).read_bytes() for name in ['metrics.jsonl','replay_ids.json','ap_matrix.npy','amount_recall_matrix.npy']}
 assert all(report[prefix+'_repeat'].values())
for pair in [('pocl-full','pocl-full-repeat')]:
 a=r/pair[0];b=r/pair[1];report['pocl_repeat']=(a/'daily.json').read_bytes()==(b/'daily.json').read_bytes();assert report['pocl_repeat']
rows=[]
for directory in sorted(r.glob('final-*')):
 if not directory.is_dir():continue
 c=json.loads((directory/'config.json').read_text());splits=np.load(directory/'splits.npz');train=set();test=set();val=set()
 for k in splits:
  target=train if k.startswith('train') else (test if k.startswith('test') else val);target.update(splits[k].tolist())
 assert not (train & test or train & val or test & val)
 assert len((directory/'metrics.jsonl').read_text().splitlines())==12
 for line in (directory/'metrics.jsonl').read_text().splitlines():
  row=json.loads(line);assert row['memory_size']<=c['memory'];assert row['replayed_total']<=c['epochs']*c['replay_budget']
 for event in json.loads((directory/'replay_ids.json').read_text()):
  assert len(event['ids'])==len(set(event['ids']));assert set(event['ids'])<=train
  assert not set(event['ids']) & (val|test)
 if 'repeat' not in directory.name:rows.append(json.loads((directory/'summary.json').read_text()))
report['split_and_budget_checks']='passed all 16 final runs'
pd.DataFrame(rows).to_csv(r/'final-summary.csv',index=False)
# Verify a saved final checkpoint reproduces a reported held-out metric.
a=r/'final-combined-17';ck=torch.load(a/'checkpoint-11.pt',weights_only=False,map_location='cpu');d=pd.read_csv(ck['config']['data']).sort_values(['TX_TIME_SECONDS','TRANSACTION_ID']).reset_index(drop=True)
h=(d.TX_TIME_SECONDS%86400)/3600
raw=np.column_stack([np.log1p(d.TX_AMOUNT),np.sin(h*2*np.pi/24),np.cos(h*2*np.pi/24),(d.TX_TIME_DAYS%7>=5).astype(float)])
x=torch.tensor((raw-np.array(ck['manifest']['scaler_mean']))/np.array(ck['manifest']['scaler_scale']),dtype=torch.float32)
model=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.ReLU(),torch.nn.Linear(32,1));model.load_state_dict(ck['model']);model.eval()
idx=d.TRANSACTION_ID.isin(np.load(a/'splits.npz')['test_11']).to_numpy()
with torch.no_grad():p=torch.sigmoid(model(x[idx]).flatten()).numpy()
m=metrics(d.TX_FRAUD.to_numpy()[idx],p,d.TX_AMOUNT.to_numpy()[idx],ck['threshold']);expected=json.loads((a/'metrics.jsonl').read_text().splitlines()[-1])['post_update']
assert m==expected;report['checkpoint_metric_recovery']='exact match'
(r/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
