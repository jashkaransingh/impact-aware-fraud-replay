"""Run the three predeclared development candidates, retaining every outcome.

Each candidate first runs an isolated current/replay optimizer smoke. Failed
smokes never proceed to full training. Never overwrites any output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ('combined-alpha075', ['--alpha', '0.75']),
    ('combined-mix050', ['--random-fraction', '0.5']),
    ('combined-classnorm', ['--score-normalization', 'class']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    env.setdefault('MPLCONFIGDIR', '/tmp/impact-mpl')
    env.setdefault('XDG_CACHE_HOME', '/tmp/impact-cache')
    records = [{'name': name, 'options': options, 'status': 'planned'} for name, options in CANDIDATES]
    sources = ['scripts/run_ieee_balanced_replay.py', 'scripts/run_ieee_no_replay_baseline.py',
               'scripts/run_ieee_policy_candidates.py', 'src/impact_replay/replay.py']
    registry = {'scope': 'Development only; three prespecified candidates; reserved evaluation unused',
                'base_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                'attempts': records}
    def save():
        (output / 'registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    save()
    for record in records:
        name, options = record['name'], record['options']
        command = [sys.executable, str(ROOT / 'scripts/run_ieee_balanced_replay.py'),
                   '--policy', 'combined', '--output', str(output / name), *options]
        record['command'] = command
        for phase, flags in [('smoke', ['--smoke-only']), ('full', [])]:
            record['status'] = f'{phase}_running'; save()
            start = time.perf_counter()
            with (output / f'{name}-{phase}.log').open('x') as log:
                completed = subprocess.run(command + flags, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record[phase] = {'returncode': completed.returncode,
                             'wall_seconds': time.perf_counter()-start,
                             'log': str(output / f'{name}-{phase}.log')}
            if completed.returncode:
                record['status'] = f'{phase}_failed'; save()
                print(f'{name}: {phase} FAILED; retained log', flush=True)
                break
            record['status'] = f'{phase}_passed'; save()
            print(f'{name}: {phase} passed ({record[phase]["wall_seconds"]:.2f}s)', flush=True)
    if any(record['status'] != 'full_passed' for record in records):
        raise SystemExit('Some candidates failed; see registry.json. No retries or replacements performed.')


if __name__ == '__main__':
    main()
