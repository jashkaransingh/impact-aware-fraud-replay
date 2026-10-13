import json,zipfile
from pathlib import Path
import numpy as np,torch
from impact_replay.controlled import evaluate
import argparse
p=argparse.ArgumentParser();p.add_argument('--prefix',choices=['test','tight'],default='test');a=p.parse_args();prefix=a.prefix;seeds=[201,202,203,204,205] if prefix=='test' else [301,302,303,304,305]
r=Path('controlled');dirs=sorted(p for p in r.glob(prefix+'-*') if p.is_dir() and 'validation' not in p.name);assert len(dirs)==(76 if prefix=='test' else 75)
report={'accepted_runs':75,'repeat_runs':int(prefix=='test'),'checks':[]}
for p in dirs:
 c=json.loads((p/'config.json').read_text());h=json.loads((p/'history.json').read_text());events=json.loads((p/'replay.json').read_text());assert len(h)==9 and len(events)==45
 z=np.load(p/'data.npz');train=set();test=set()
 for t in range(9):
  ids=z[f'train_{t}_ids'];assert not train.intersection(ids.tolist());train.update(ids.tolist())
 for t in range(2):test.update(z[f'test_{t}_ids'].tolist())
 assert not train & test
 for e in events:
  assert len(e['ids'])==len(set(e['ids']));assert len(e['ids'])<=c['replay_budget'];assert set(e['ids'])<=train;assert not set(e['ids'])&test
  assert all(i<e['window']*c['train_per_window'] for i in e['ids'])
 for t,row in enumerate(h):
  assert row['memory']<=c['memory'];assert row['replayed']<=c['epochs']*c['replay_budget'];assert row['steps']==80
  assert zipfile.is_zipfile(p/f'checkpoint-{t}.pt')
 ck=torch.load(p/'checkpoint-8.pt',weights_only=False,map_location='cpu');assert len(ck['memory'])<=c['memory'];assert len({q['id'] for q in ck['memory']})==len(ck['memory'])
report['checks'].append('all run histories, checkpoint containers, steps, memory and replay budgets, temporal eligibility and split disjointness passed')
a=r/f'{prefix}-independent-combined-{seeds[0]}'
if prefix=='test':
    b=r/'test-independent-combined-201-repeat'
    for n in ['history.json','replay.json']:
     assert (a/n).read_bytes()==(b/n).read_bytes()
    s1=json.loads((a/'summary.json').read_text());s2=json.loads((b/'summary.json').read_text());s1.pop('seconds');s2.pop('seconds');assert s1==s2
    report['checks'].append('identical repeated histories, selected IDs, and all summary metrics')
for seed in seeds:
 for policy in ['none','random','forgetting']:
  summaries=[json.loads((r/f'{prefix}-{mode}-{policy}-{seed}'/'summary.json').read_text()) for mode in ['independent','a_high','b_high']]
  for k in ['A_learned_ap','A_after_B_ap','A_forgetting','B_learned_ap','A_return_ap','B_after_A_ap','B_forgetting']:
   assert summaries[0][k]==summaries[1][k]==summaries[2][k]
 for mode in ['independent','a_high','b_high']:
  paths=[r/f'{prefix}-{mode}-{policy}-{seed}' for policy in ['none','random','forgetting','impact','combined']]
  assert len({json.loads((p/'provenance.json').read_text())['data_sha256'] for p in paths})==1
  ids=[sorted(q['id'] for q in torch.load(p/'checkpoint-8.pt',weights_only=False)['memory']) for p in paths]
  assert all(i==ids[0] for i in ids)
report['checks'].append('identical data and buffer admission across policies; amount changes do not change non-impact policy AP')
ck=torch.load(a/'checkpoint-5.pt',weights_only=False);c=ck['config'];m=torch.nn.Sequential(torch.nn.Linear(4,c['hidden']),torch.nn.ReLU(),torch.nn.Linear(c['hidden'],1));m.load_state_dict(ck['model']);z=np.load(a/'data.npz');torch.set_num_threads(1)
h=json.loads((a/'history.json').read_text())
for i in [0,1]:
 block={k:z[f'test_{i}_{k}'] for k in ['x','y','amount','ids']};assert evaluate(m,block)==h[5]['after'][i]
report['checks'].append('checkpoint-5 reproduces both domain metric dictionaries exactly')
(r/f'verification-{prefix}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
