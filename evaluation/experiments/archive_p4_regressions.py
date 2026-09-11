"""Seal P4 diagnostic evidence, including explicitly incomplete Agent attempts."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from arkb.evaluation.external import digest, verify_checksums, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    sidecar=args.output.with_suffix('.manifest.json')
    if any(p.exists() for p in (args.output,sidecar,args.audit_output)):
        raise ValueError('Retain existing archives and audits; use new paths.')
    roots={
        'regression':repo/'evaluation/results/p4-regression',
        'live-freshness':repo/'evaluation/results/p4-freshness-live',
        'incomplete-agent-v1':repo/'evaluation/results/p4-bright-stackoverflow-agent',
        'incomplete-musique-v2':repo/'evaluation/results/p4-musique-agent-v2',
        'measured-source/src':repo/'src',
    }
    for name in ('regression','live-freshness','incomplete-agent-v1','incomplete-musique-v2'):verify_checksums(roots[name])
    if json.loads((roots['incomplete-agent-v1']/'experiment.json').read_text())['status']!='failed':
        raise ValueError('Expected the preserved failed v1 attempt.')
    if json.loads((roots['incomplete-musique-v2']/'experiment.json').read_text())['status']!='failed':
        raise ValueError('Expected the preserved failed MuSiQue v2 attempt.')
    files={}
    for prefix,root in roots.items():
        for p in root.rglob('*'):
            # A sealed fixture may bind WAL/SHM bytes in its checksums. Preserve
            # those files as evidence instead of silently omitting them.
            if p.is_file() and not p.name.endswith('.pyc') and '__pycache__' not in p.parts:
                if p.is_symlink():raise ValueError('Cannot archive symlinks.')
                files[f'{prefix}/{p.relative_to(root).as_posix()}']=p
    for p in (repo/'evaluation/audits').glob('20260910-p4-*'):
        if p.is_file():files['audits/'+p.name]=p
    for p in (repo/'evaluation/audits').glob('*p4*.py'):
        files['audits/'+p.name]=p
    for n in ('pyproject.toml','uv.lock'):files['measured-source/'+n]=repo/n
    for n in ('p4_regression.py','archive_p4_regressions.py','check_p4_deterministic.py','summarize_p4_costs.py','p4_status.py','p4_archive_parts.py'):
        files['experiments/'+n]=Path(__file__).with_name(n)
    manifest={'schema':'arkb-p4-regression-evidence-v1','status':'sealed-diagnostic-evidence',
        'scope':'Retained regression results, fixtures, source and incomplete Bright v1/MuSiQue v2 attempts. Sealing does not convert an incomplete attempt into a completed experiment.',
        'release_eligible':False,'files':{n:digest(p) for n,p in sorted(files.items())}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'x:gz') as archive:
        for n,p in sorted(files.items()):archive.add(p,arcname=n,recursive=False)
        body=(json.dumps(manifest,indent=2)+'\n').encode()
        info=tarfile.TarInfo('archive-manifest.json');info.size=len(body);archive.addfile(info,io.BytesIO(body))
    record={'archive':args.output.name,'sha256':digest(args.output),'bytes':args.output.stat().st_size,
            'files':len(files)+1,'status':manifest['status'],'scope':manifest['scope'],'release_eligible':False}
    write_json(sidecar,record)
    with tempfile.TemporaryDirectory(prefix='arkb-p4-regression-archive-') as directory:
        root=Path(directory).resolve()
        with tarfile.open(args.output,'r:gz') as archive:
            members=archive.getmembers()
            if len({m.name for m in members})!=len(members) or any(
                    not m.isfile() or not (root/m.name).resolve().is_relative_to(root) for m in members):
                raise ValueError('Unsafe archive member.')
            archive.extractall(root,filter='data')
        retained=json.loads((root/'archive-manifest.json').read_text())
        if retained!=manifest:raise ValueError('Archive manifest changed.')
        if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(files)|{'archive-manifest.json'}:
            raise ValueError('Archive inventory changed.')
        for n,h in retained['files'].items():
            if digest(root/n)!=h:raise ValueError('Archive member changed.')
        for prefix in ('regression','live-freshness','incomplete-agent-v1','incomplete-musique-v2'):verify_checksums(root/prefix)
        subprocess.run([sys.executable,str(root/'experiments/p4_regression.py'),'probe','--output',str(root/'replayed')],
            env={**os.environ,'PYTHONPATH':str(root/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1'},
            check=True,capture_output=True,text=True)
        replay=json.loads((root/'replayed/freshness.json').read_text())
        if replay!=json.loads((root/'regression/freshness.json').read_text()):
            raise ValueError('Frozen-code freshness replay differs.')
    write_json(args.audit_output,{'passed':True,'archive_sha256':record['sha256'],
        'verified_files':len(files)+1,'deterministic_freshness_checks_replayed':replay['passed'],
        'live_model_probe':'Original results and hashes retained, not rerun.',
        'incomplete_agent_attempts':{'bright_v1':'Original failed status and artifacts preserved; no score imputed.',
            'musique_v2':'One completed variant and second-context setup failure retained; zero complete pairs. No output is selected against the v3 restart.'},
        'release_eligible':False})
    print(json.dumps(record))


if __name__=='__main__':main()
