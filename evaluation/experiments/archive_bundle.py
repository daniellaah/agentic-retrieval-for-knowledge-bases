"""Archive retained experiment files; refuse overwrites and incomplete runs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bundle=a.bundle.resolve()
    metadata=bundle/('experiment.json' if (bundle/'experiment.json').exists() else 'run_metadata.json')
    if json.loads(metadata.read_text())['status']!='completed':
        raise ValueError('Only completed experiments can use this publication archive path.')
    if a.output.exists() or a.output.with_suffix('.manifest.json').exists():
        raise ValueError('Archive already exists.')
    expected=json.loads((bundle/'checksums.json').read_text())
    for name,digest in expected.items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Invalid artifact: {name}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'x:gz') as archive:
        for name in sorted([*expected,'checksums.json']):
            archive.add(bundle/name,arcname=f'{bundle.name}/{name}',recursive=False)
    manifest={'archive':a.output.name,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
              'bytes':a.output.stat().st_size,'root':bundle.name,'files':len(expected)+1,
              'source_bundle':str(bundle),'status':'completed',
              'includes':'Frozen source, lockfile, cases/corpus, raw observations, index, metadata, scores and hashes; excludes transient WAL/SHM/cache and execution log.'}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
