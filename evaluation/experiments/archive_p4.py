"""Archive completed P4 score-replay inputs, retaining corpus text and provenance."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from arkb.evaluation.external import digest,load_external,verify_checksums,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path('evaluation/results/p4-inputs'))
    a=p.parse_args();sidecar=a.output.with_suffix('.manifest.json')
    if a.output.exists() or sidecar.exists():raise ValueError('Archive path already exists.')
    files={};omitted={};runs=[]
    if a.dataset:
        load_external(a.dataset)
        files.update({'dataset/'+n:a.dataset/n for n in ('manifest.json','corpus.jsonl','queries.json','qrels.json','aspects.json')})
    for run in a.run:
        if json.loads((run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete experiment: '+str(run))
        verify_checksums(run);checks=json.loads((run/'checksums.json').read_text());runs.append(run.name)
        for name,h in checks.items():
            target=f'runs/{run.name}/{name}'
            if name.endswith('.sqlite'):
                omitted[target]={'sha256':h,'bytes':(run/name).stat().st_size,'reason':'Model/index rerun cache. Score replay uses original candidate records and full source text.'}
            else:files[target]=run/name
        files[f'runs/{run.name}/checksums.json']=run/'checksums.json'
        if (run/'execution.log').exists():files[f'runs/{run.name}/execution.log']=run/'execution.log'
    for path in a.inputs.rglob('*'):
        if path.is_file() and (path.suffix in ('.md','.py') or path.name in ('LICENSE','downloads.json','agentic_sample_ids.json')):
            files['public-inputs/'+path.relative_to(a.inputs).as_posix()]=path
    repo=Path(__file__).resolve().parents[2]
    for name in ('replay_p4.py','replay_p4_agents.py','verify_p4.py','audit_p4_rerank.py','check_p4_musique.py','verify_p4_archive.py'):
        files['audits/'+name]=repo/'evaluation/audits'/name
    files['experiments/fetch_p4.py']=Path(__file__).with_name('fetch_p4.py')
    files['experiments/prepare_p4.py']=Path(__file__).with_name('prepare_p4.py')
    files['experiments/p4-downloads.lock.json']=Path(__file__).with_name('p4-downloads.lock.json')
    files['experiments/archive_p4.py']=Path(__file__)
    files['experiments/summarize_p4_agents.py']=Path(__file__).with_name('summarize_p4_agents.py')
    manifest={'schema':'arkb-p4-score-archive-v1','runs':runs,
        'scope':'Offline score and ranking-assembly replay. Includes full normalized corpus, frozen measured source, raw candidates/observations and official metric source. Model weights and SQLite/Qdrant caches are omitted; generating new model outputs requires rebuilding them.',
        'release_eligible':False,'omitted':omitted,'files':{n:digest(path) for n,path in sorted(files.items())}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'x:gz',compresslevel=6) as archive:
        for name,path in sorted(files.items()):
            if path.is_symlink():raise ValueError('Do not archive symlinks.')
            archive.add(path,arcname=name,recursive=False)
        body=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        member=tarfile.TarInfo('archive-manifest.json');member.size=len(body);member.mode=0o644
        archive.addfile(member,io.BytesIO(body))
    record={'archive':a.output.name,'sha256':digest(a.output),'bytes':a.output.stat().st_size,
            'files':len(files)+1,'runs':runs,'omitted_files':len(omitted),'scope':manifest['scope'],'release_eligible':False}
    write_json(sidecar,record);print(json.dumps(record))


if __name__=='__main__':main()
