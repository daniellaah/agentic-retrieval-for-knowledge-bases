"""Run deterministic regression probes, or reserve a reviewed release holdout."""
import argparse
from pathlib import Path
from arkb.evaluation.external import load_external, write_json
from arkb.evaluation.regression import freshness_probe, scale_probe, reserve_core_release


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    probe=commands.add_parser('probe');probe.add_argument('--output',type=Path,required=True)
    probe.add_argument('--scale-dataset',type=Path)
    reserve=commands.add_parser('reserve');reserve.add_argument('--core',type=Path,required=True)
    reserve.add_argument('--ledger',type=Path,required=True);reserve.add_argument('--release-id',required=True)
    args=parser.parse_args()
    if args.command=='reserve':
        event=reserve_core_release(args.core,args.ledger,args.release_id);print(event['sha256']);return
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'freshness.json',freshness_probe(args.output/'freshness-corpus'))
    if args.scale_dataset:write_json(args.output/'scale.json',scale_probe(load_external(args.scale_dataset)))
    print('Deterministic regression probes completed.')


if __name__=='__main__':main()
