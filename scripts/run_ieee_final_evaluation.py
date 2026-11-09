"""Metadata-only readiness/freeze, followed by explicit future inference stages."""
import argparse
from pathlib import Path

from ieee_multiseed import ROOT, read, write
from ieee_final_evaluation import build_contract, execute_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Verify development artifacts only; never open raw source')
    freeze = sub.add_parser('freeze', help='Write a new metadata-only immutable contract')
    freeze.add_argument('--output', required=True, type=Path)
    for phase in ('validation', 'reserved'):
        command = sub.add_parser(phase, help=f'Explicitly read and score {phase} records; no training')
        command.add_argument('--contract', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
        if phase == 'reserved':
            command.add_argument('--validation', required=True, type=Path)
    args = parser.parse_args()
    if args.command in ('check', 'freeze'):
        contract = build_contract(ROOT/'results/ieee-preparation/preparation.json',
                                  ROOT/'results/ieee-preparation/windows',
                                  ROOT/'results/ieee-multiseed-501-505/registry.json')
        if args.command == 'freeze':
            args.output.mkdir(parents=True, exist_ok=False)
            write(args.output/'contract.json', contract)
        print(f'Ready: {len(contract["models"])} frozen development checkpoints; no raw source opened. Contract {contract["contract_sha256"]}')
        return
    result = execute_phase(read(args.contract), args.command, args.output, getattr(args, 'validation', None))
    print(f'Completed {args.command}: {len(result["per_seed"])} method/seed metrics saved in {args.output}')


if __name__ == '__main__':
    main()
