"""Inspect intake readiness or freeze a reviewed, grouped core dataset."""
import argparse
import json
from pathlib import Path
from arkb.evaluation.core import core_readiness, freeze_core
from arkb.evaluation.v2 import load_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--freeze-to',type=Path)
    a=p.parse_args()
    if a.report.exists(): raise ValueError('Report exists; choose a new audit path.')
    registry=json.loads(a.registry.read_text());policy=json.loads(a.policy.read_text())
    data=load_dataset(a.dataset,notes_dir=a.dataset/'corpus',allow_provisional=True)
    report=core_readiness(data,registry,policy,directory=a.dataset)
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if a.freeze_to: freeze_core(a.dataset,registry,policy,a.freeze_to)
    print(json.dumps({k:v for k,v in report.items() if k not in ('problems','split_plan')},ensure_ascii=False))


if __name__=='__main__':main()
