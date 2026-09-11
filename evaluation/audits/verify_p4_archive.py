"""Verify an archive hash, extract safely and replay each retained retrieval run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from arkb.evaluation.external import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();record=json.loads(a.archive.with_suffix('.manifest.json').read_text())
    if digest(a.archive)!=record['sha256'] or a.archive.stat().st_size!=record['bytes']:raise ValueError('Archive digest mismatch.')
    results=[]
    with tempfile.TemporaryDirectory(prefix='arkb-p4-replay-') as directory:
        root=Path(directory).resolve()
        with tarfile.open(a.archive,'r:gz') as archive:
            members=archive.getmembers()
            if len({m.name for m in members})!=len(members):raise ValueError('Duplicate archive members.')
            if any(not (root/m.name).resolve().is_relative_to(root) or not m.isfile() for m in members):
                raise ValueError('Unsafe archive member.')
            archive.extractall(root,filter='data')
        manifest=json.loads((root/'archive-manifest.json').read_text())
        if manifest['schema']!='arkb-p4-score-archive-v1':raise ValueError('Unknown archive schema.')
        if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(manifest['files'])|{'archive-manifest.json'}:
            raise ValueError('Archive file inventory differs.')
        for name,h in manifest['files'].items():
            if digest(root/name)!=h:raise ValueError('Archive member checksum mismatch.')
        for name in manifest['runs']:
            run=root/'runs'/name
            for file,h in json.loads((run/'checksums.json').read_text()).items():
                key=f'runs/{name}/{file}'
                if key in manifest['omitted']:
                    if manifest['omitted'][key]['sha256']!=h:raise ValueError('Omitted cache registration mismatch.')
                elif digest(run/file)!=h:raise ValueError('Measured artifact differs.')
            agent=(run/'run_p4_agents.py').exists()
            command=[sys.executable,str(root/'audits'/('replay_p4_agents.py' if agent else 'replay_p4.py')),str(run),'--archive-root',str(root)]
            if (root/'dataset').exists():command+=['--dataset',str(root/'dataset')]
            result=subprocess.run(command,text=True,capture_output=True,check=True)
            results.append({'run':name,**json.loads(result.stdout)})
    write_json(a.output,{'archive_sha256':record['sha256'],'replayed':results,'passed':True,
        'scope':'No model or Qdrant calls; metric and ranking assembly integrity, not annotation truth.','release_eligible':False})
    print(json.dumps(results))


if __name__=='__main__':main()
