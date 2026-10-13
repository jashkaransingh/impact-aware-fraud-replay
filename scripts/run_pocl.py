"""Execute preserved POCL source with explicit, recorded schedule overrides."""
import argparse, os, sys, json, random, hashlib, time
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser()
p.add_argument('--pretrain-epochs',type=int,default=1)
p.add_argument('--offline-epochs',type=int,default=1)
p.add_argument('--online-steps',type=int,default=1)
p.add_argument('--days',type=int,default=18)
p.add_argument('--seed',type=int,default=17)
p.add_argument('--out',required=True)
p.add_argument('--variant',choices=['upstream','audited'],default='upstream')
a=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=(root/a.out).resolve();out.mkdir(parents=True,exist_ok=False)
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
os.environ['MPLBACKEND']='Agg'
source_dir=root/('upstream/POCL' if a.variant=='upstream' else 'work/pocl-audited')
sys.path.insert(0,str(source_dir))
import models,tool
work=root/'work/pocl';tool.data_route=str(work/'Dataset')+'/'
source=(source_dir/'main.py').read_text()
source=source.replace('online_dataset = get_online_dataset()',f'online_dataset = get_online_dataset()[:{a.days}]')
source=source.replace('pretrain(pretrain_model, pretrain_days, 300, optimizer_pretrain)',f'pretrain(pretrain_model, pretrain_days, {a.pretrain_epochs}, optimizer_pretrain)')
source=source.replace('train_offline_model(100, pretrain_days, train_optimizer)',f'train_offline_model({a.offline_epochs}, pretrain_days, train_optimizer)')
source=source.replace('range(10)',f'range({a.online_steps})')
(out/'executed_main.py').write_text(source)
(out/'config.json').write_text(json.dumps(vars(a),indent=2))
os.chdir(out); namespace={};start=time.time()
exec(compile(source,'upstream/main.py [schedule overrides]','exec'),namespace)
metrics={k:float(np.mean(namespace[k])) for k in ['aucu','f1u','accu']}
metrics.update(variant=a.variant,seconds=time.time()-start,online_days=len(namespace['aucu']),seed=a.seed,status=('full_released_schedule' if (a.pretrain_epochs,a.offline_epochs,a.online_steps,a.days)==(300,100,10,364) else 'reduced_schedule_upstream_behavior'))
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
(out/'daily.json').write_text(json.dumps({k:namespace[k] for k in ['aucu','f1u','accu']},indent=2))
torch.save({'encoder':namespace['pretrain_model'].state_dict(),'classifier':namespace['classifier'].state_dict(),'config':vars(a)},out/'checkpoint.pt')
print(json.dumps(metrics))
