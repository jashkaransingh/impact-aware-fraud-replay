from pathlib import Path
import json
from impact_replay.controlled import run,joint_control
c=json.loads(Path('configs/controlled-validation.json').read_text())
run(dict(c,seed=102),'controlled/validation-none-102')
for seed in [101,102]:
 joint_control(dict(c,seed=seed),f'controlled/validation-joint-{seed}')
 run(dict(c,seed=seed,policy='random'),f'controlled/validation-random-{seed}')
gate=[]
for seed in [101,102]:
 n=json.loads(Path(f'controlled/validation-none-{seed}/summary.json').read_text());j=json.loads(Path(f'controlled/validation-joint-{seed}/summary.json').read_text())
 g={'seed':seed,'initial_A_AP_at_least_080':n['A_learned_ap']>=.8,'A_AP_drop_at_least_010':n['A_forgetting']>=.1,'current_B_AP_at_least_080':n['B_learned_ap']>=.8,'joint_both_AP_at_least_090':all(m['ap']>=.9 for m in j['domain_metrics'])};gate.append(g)
Path('controlled/validation-gates.json').write_text(json.dumps(gate,indent=2));assert all(all(v for k,v in g.items() if k!='seed') for g in gate);print('ALL PRESPECIFIED GATES PASSED')
