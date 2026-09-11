"""Check that all retained P4 attempts used identical production Python sources."""
import argparse
import json
from pathlib import Path

from arkb.evaluation.external import digest, verify_checksums, write_json


def production_files(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*.py'))
            if 'evaluation' not in p.relative_to(root).parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Retain previous audit results; use a new output path.')
    current = production_files(Path('src/arkb'))
    if not current:
        raise ValueError('Run from the repository root.')
    runs = {}
    for run in args.run:
        verify_checksums(run)
        metadata = json.loads((run / 'experiment.json').read_text())
        if metadata['status'] not in ('completed', 'failed'):
            raise ValueError('Attempt is still running: ' + str(run))
        if production_files(run / 'measured-source/src/arkb') != current:
            raise ValueError('Production source inventory or bytes differ: ' + str(run))
        runs[str(run)] = {'status': metadata['status'], 'identical_production_files': len(current),
                          'experiment_sha256': digest(run / 'experiment.json'),
                          'checksums_sha256': digest(run / 'checksums.json')}
    write_json(args.output, {'passed': True, 'files': current, 'runs': runs,
        'audit_sha256': digest(Path(__file__)),
        'scope': 'Exact production Python source inventory and byte identity across current source and retained terminal attempts. Evaluation instrumentation is excluded; the original failed attempt remains failed.',
        'release_eligible': False})
    print(json.dumps({'passed': True, 'attempts': len(runs), 'production_files': len(current)}))


if __name__ == '__main__':
    main()
