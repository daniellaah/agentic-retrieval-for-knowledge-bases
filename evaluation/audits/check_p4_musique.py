"""Cross-check paired MuSiQue metrics against inspected hash-locked official code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from arkb.evaluation.external import digest,read_jsonl,write_json
from arkb.evaluation.multihop import musique_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path('evaluation/results/p4-inputs'))
    p.add_argument('--lock',type=Path,default=Path('evaluation/experiments/p4-downloads.lock.json'))
    a=p.parse_args();lock=json.loads(a.lock.read_text());registered={n:v for n,v in lock['files'].items() if n.startswith('official/musique/')}
    if len(registered)!=7:raise ValueError('Official scoring source inventory changed; review before execution.')
    if {p.relative_to(a.inputs).as_posix() for p in (a.inputs/'official/musique').rglob('*.py')}!=set(registered):
        raise ValueError('Unexpected official scoring module.')
    for n,v in registered.items():
        if digest(a.inputs/n)!=v['sha256']:raise ValueError('Official scoring source checksum mismatch.')
    gold=read_jsonl(a.gold);predictions=read_jsonl(a.predictions);independent=musique_metrics(gold,predictions)
    with tempfile.TemporaryDirectory(prefix='arkb-musique-score-') as directory:
        result=Path(directory)/'official.json'
        subprocess.run([sys.executable,str(a.inputs/'official/musique/evaluate_v1.0.py'),str(a.predictions),str(a.gold),
                        '--output_filepath',str(result)],capture_output=True,text=True,check=True)
        official=json.loads(result.read_text())
    if any(round(independent[k],3)!=v for k,v in official.items()):raise ValueError('Official MuSiQue scorer differs.')
    strata={}
    for hop in (2,3,4):
        idx=[i for i,g in enumerate(gold) if g['id'].startswith(f'{hop}hop')]
        if idx:strata[str(hop)]=musique_metrics([gold[i] for i in idx],[predictions[i] for i in idx])
    confusion=Counter(f"gold={g['answerable']},predicted={p['predicted_answerable']}" for g,p in zip(gold,predictions))
    write_json(a.output,{'official':official,'independent':independent,'by_hops':strata,
        'answerability_confusion':dict(confusion),'match_at_official_precision':True,
        'predictions_sha256':digest(a.predictions),'gold_sha256':digest(a.gold),
        'official_source_hashes':{n:v['sha256'] for n,v in registered.items()},'release_eligible':False})
    print(json.dumps({'official':official,'match':True}))


if __name__=='__main__':main()
