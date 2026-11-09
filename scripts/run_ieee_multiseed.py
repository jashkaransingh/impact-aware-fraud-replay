"""Execute only the prespecified five-method, five-seed development study.

Seed 501 is reused. Twenty new full runs are made in a new output directory.
A failed run stops the study; no seeds/settings are replaced automatically.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ieee_multiseed import (ROOT, SEEDS, METHODS, REUSED, SOURCE_FILES, audit_seed,
                           hashes, preparation_manifest, read, run_method, sha,
                           verify_hashes, write)


def verify_parent():
    parent = read(ROOT/'summaries/ieee-policy-study-501/final-reproduction-audit.json')
    verify_hashes({str(ROOT/path): value for path, value in parent['protected_anchor_sha256'].items()})
    if sha(ROOT/'scripts/run_ieee_balanced_replay.py') != parent['current_runner_sha256']:
        raise AssertionError('Verified legacy replay source changed')
    report = read(ROOT/'summaries/ieee-policy-study-501/final-comparison.json')
    if len(report['rows']) != 8 or not all(all(checks.values()) for checks in report['audits'].values()):
        raise AssertionError('Parent eight-method audits failed')
    for method, source in report['provenance'].items():
        if sha(Path(source['path'])/'results.json') != source['results_sha256']:
            raise AssertionError(f'Parent result changed: {method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-single', choices=METHODS)
    parser.add_argument('--seed', choices=SEEDS, type=int)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.run_single:
        if args.seed is None or args.seed == 501:
            parser.error('Single runs require a new training seed (502–505); reuse seed 501')
        run_method(args.run_single, args.seed, args.output, args.baseline)
        return
    if args.seed is not None or args.baseline is not None:
        parser.error('seed/baseline are only valid with --run-single')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite study: {output}')
    verify_parent()
    manifest = preparation_manifest()
    old_roots = [ROOT/'results/ieee-no-replay-baseline-501', ROOT/'results/ieee-class-balanced-replay-501',
                 ROOT/'results/ieee-policy-study-501']
    protected = hashes([p for root in old_roots for p in root.rglob('*') if p.is_file()])
    reused_audits = audit_seed(REUSED, 501, manifest)
    output.mkdir(parents=True, exist_ok=False)
    records = [{'seed': seed, 'method': method, 'status': 'reused' if seed == 501 else 'planned',
                'path': str(REUSED[method] if seed == 501 else output/f'seed-{seed}'/method)}
               for seed in SEEDS for method in METHODS]
    registry = {'seeds': list(SEEDS), 'methods': list(METHODS), 'data_split_seed': 501,
                'primary_question': 'Classnorm minus original combined historical amount recall at 5%',
                'protocol_sha256': sha(ROOT/'docs/ieee-multiseed-protocol.md'),
                'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': {name: sha(ROOT/name) for name in SOURCE_FILES},
                'prepared_data': manifest, 'protected_existing_artifacts': protected,
                'runs': records, 'seed_audits': {'501': reused_audits}, 'status': 'running'}
    write(output/'registry.json', registry)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    env.setdefault('MPLCONFIGDIR', '/tmp/impact-mpl')
    env.setdefault('XDG_CACHE_HOME', '/tmp/impact-cache')
    for seed in SEEDS[1:]:
        for method in METHODS:
            record = next(r for r in records if r['seed'] == seed and r['method'] == method)
            command = [sys.executable, str(Path(__file__).resolve()), '--run-single', method,
                       '--seed', str(seed), '--output', record['path']]
            if method != 'no-replay':
                command += ['--baseline', str(output/f'seed-{seed}'/'no-replay')]
            record.update(command=command, status='running')
            write(output/'registry.json', registry)
            start = time.perf_counter()
            with (output/f'seed-{seed}-{method}.log').open('x') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, wall_seconds=time.perf_counter()-start,
                          status='passed' if result.returncode == 0 else 'failed')
            write(output/'registry.json', registry)
            print(f'{seed} {method}: {record["status"]} ({record["wall_seconds"]:.2f}s)', flush=True)
            if result.returncode:
                registry['status'] = 'failed'
                write(output/'registry.json', registry)
                raise SystemExit('Study stopped after failed run; outputs retained, no replacements attempted')
        paths = {method: output/f'seed-{seed}'/method for method in METHODS}
        registry['seed_audits'][str(seed)] = audit_seed(paths, seed, manifest)
        verify_hashes(manifest['files_sha256'])
        verify_hashes(protected)
        write(output/'registry.json', registry)
        print(f'{seed}: paired audit and immutable-input checks passed', flush=True)
    registry.update(status='complete', prior_outputs_unchanged=True, prepared_data_unchanged=True)
    write(output/'registry.json', registry)


if __name__ == '__main__':
    main()
