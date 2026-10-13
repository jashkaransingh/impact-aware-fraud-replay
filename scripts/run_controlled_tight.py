import json
from pathlib import Path
from impact_replay.controlled import run
c=json.loads(Path('configs/controlled-validation.json').read_text());c.update(memory=64,replay_budget=32)
run(dict(c,seed=103,policy='combined'), 'controlled/tight-validation-combined-103')
c['evaluation_kind']='held_out_budget_robustness'
for seed in [301,302,303,304,305]:
 for mode in ['independent','a_high','b_high']:
  for policy in ['none','random','forgetting','impact','combined']:
   run(dict(c,seed=seed,amount_mode=mode,policy=policy),f'controlled/tight-{mode}-{policy}-{seed}')
