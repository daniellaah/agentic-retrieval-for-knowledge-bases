"""Blind-order exact unique P3 output packets; keep arm keys in a separate file."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from arkb.evaluation.outputs import pending_review,verify_packet


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if json.loads((a.bundle/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete experiment.')
    rows=[json.loads(line) for line in (a.bundle/'agent-results.jsonl').read_text().splitlines()]
    packets={};keys=[]
    for row in rows:
        packet=row['packet'];verify_packet(packet);sha=packet['packet_sha256']
        if sha in packets and packets[sha]!=packet:raise ValueError('Conflicting packet hash.')
        packets[sha]=packet
        keys.append({'packet_sha256':sha,'case_id':row['case_id'],'arm':row['arm'],'trial':row['trial']})
    shuffled=list(packets);random.Random(20260910).shuffle(shuffled)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'blind-packets.json',[{'packet':packets[sha],'review':pending_review(packets[sha])} for sha in shuffled])
    write(a.output/'operator-key.json',keys)
    write(a.output/'progress.json',{'trial_outputs':len(rows),'unique_packets':len(packets),'human_reviewed_packets':0,'status':'pending',
        'note':'Identical packet hashes may share an exact-content review; this does not create new independent cases or families.'})
    write(a.output/'manifest.json',{'schema_version':'arkb-p3-review-export-v1','bundle':a.bundle.name,
        'raw_agent_sha256':hashlib.sha256((a.bundle/'agent-results.jsonl').read_bytes()).hexdigest(),
        'export_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir()) if p.is_file()}})
    print(json.dumps({'outputs':len(rows),'unique_packets':len(packets),'reviewed':0}))


if __name__=='__main__':main()
