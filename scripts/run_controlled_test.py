import json,hashlib
from pathlib import Path
from impact_replay.controlled import run
c=json.loads(Path('configs/controlled-validation.json').read_text());c['evaluation_kind']='held_out_test'
for seed in [201,202,203,204,205]:
 for mode in ['independent','a_high','b_high']:
  for policy in ['none','random','forgetting','impact','combined']:
   run(dict(c,seed=seed,amount_mode=mode,policy=policy),f'controlled/test-{mode}-{policy}-{seed}')
run(dict(c,seed=201,amount_mode='independent',policy='combined'),'controlled/test-independent-combined-201-repeat')
