"""Normalize the pinned public inputs and freeze query selections before running."""
import argparse
from pathlib import Path
import json
from arkb.evaluation.external import import_public, musique_groups, write_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lock',type=Path,default=Path(__file__).with_name('p4-downloads.lock.json'))
    args=parser.parse_args()
    lock=json.loads(args.lock.read_text())
    for name,expected in lock['files'].items():
        path=args.inputs/name
        if not path.resolve().is_relative_to(args.inputs.resolve()) or digest(path)!=expected['sha256']:
            raise ValueError('Input differs from pinned download lock: '+name)
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('scifact','bright-stackoverflow','bright-robotics'):
        data=import_public(args.inputs,name)
        data.save(args.output/name,materialize=name=='scifact')
        print(name,len(data.corpus),len(data.queries),flush=True)
    rows,metadata=musique_groups(args.inputs,groups=100)
    write_json(args.output/'musique-selected.json',{'metadata':metadata,'rows':rows})
    print('musique',len(rows),'variants',flush=True)


if __name__=='__main__':main()
