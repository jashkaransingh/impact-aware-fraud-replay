import json,sys
from pathlib import Path
from impact_replay.experiment import run
import argparse
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/calibrated.json');p.add_argument('--prefix',default='calibrated');a=p.parse_args()
c=json.loads(Path(a.config).read_text())
for seed in [17,29,43]:
 for policy in ['none','random','forgetting','impact','combined']:
  run(dict(c,seed=seed,policy=policy),f'results/{a.prefix}-{policy}-{seed}')
run(dict(c,seed=17,policy='combined'),f'results/{a.prefix}-combined-17-repeat')
