"""Split a sealed P4 archive into portable parts, or reassemble and fully replay it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from arkb.evaluation.external import digest, write_json


def local_file(directory, name):
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError('Expected a local artifact basename.')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular artifact file: ' + str(path))
    return path


def split(archive, output, part_mib):
    sidecar = archive.with_suffix('.manifest.json')
    record = json.loads(sidecar.read_text())
    if digest(archive) != record['sha256'] or archive.stat().st_size != record['bytes']:
        raise ValueError('Unsealed or changed source archive.')
    if output.exists() or any(output.parent.glob(archive.name + '.part-*')):
        raise ValueError('Retain previous parts; use a new output directory.')
    if output.parent.resolve() != archive.parent.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        destination = output.parent / sidecar.name
        if destination.exists():
            raise ValueError('Retain previous archive metadata.')
        shutil.copyfile(sidecar, destination)
    parts = []
    with archive.open('rb') as source:
        while chunk := source.read(part_mib * 2**20):
            path = output.parent / f'{archive.name}.part-{len(parts):03d}'
            with path.open('xb') as stream:
                stream.write(chunk)
            parts.append({'name': path.name, 'bytes': len(chunk), 'sha256': digest(path)})
    write_json(output, {'schema': 'arkb-p4-multipart-v1', 'archive': archive.name,
        'bytes': record['bytes'], 'sha256': record['sha256'],
        'archive_manifest': sidecar.name, 'archive_manifest_sha256': digest(sidecar),
        'parts': parts, 'packager_sha256': digest(Path(__file__)),
        'scope': 'Lossless transport parts of the original sealed archive; scores and experiment protocols are unchanged.',
        'release_eligible': False})
    print(json.dumps({'manifest': str(output), 'parts': len(parts), 'archive_bytes': record['bytes']}))


def replay(manifest_path, output):
    if output.exists():
        raise ValueError('Retain previous audit outputs; use a new path.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'arkb-p4-multipart-v1':
        raise ValueError('Unknown multipart schema.')
    directory = manifest_path.parent
    sidecar = local_file(directory, manifest['archive_manifest'])
    if digest(sidecar) != manifest['archive_manifest_sha256']:
        raise ValueError('Archive metadata changed.')
    record = json.loads(sidecar.read_text())
    if (record['archive'], record['bytes'], record['sha256']) != (
            manifest['archive'], manifest['bytes'], manifest['sha256']):
        raise ValueError('Archive identity differs.')
    names = [p['name'] for p in manifest['parts']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Missing or duplicated parts.')
    if Path(manifest['archive']).name != manifest['archive']:
        raise ValueError('Invalid archive basename.')
    with tempfile.TemporaryDirectory(prefix='arkb-p4-parts-') as temporary:
        archive = Path(temporary) / manifest['archive']
        with archive.open('xb') as target:
            for part in manifest['parts']:
                path = local_file(directory, part['name'])
                if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                    raise ValueError('Part digest mismatch: ' + part['name'])
                with path.open('rb') as source:
                    shutil.copyfileobj(source, target)
        if archive.stat().st_size != record['bytes'] or digest(archive) != record['sha256']:
            raise ValueError('Reassembled archive differs.')
        shutil.copyfile(sidecar, archive.with_suffix('.manifest.json'))
        audit = Path(temporary) / 'replay.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'audits/verify_p4_archive.py'),
                        str(archive), '--output', str(audit)], check=True, capture_output=True, text=True)
        result = json.loads(audit.read_text())
    result['multipart'] = {'manifest_sha256': digest(manifest_path), 'parts_verified': len(names),
                           'archive_byte_identity': True, 'packager_sha256': digest(Path(__file__))}
    write_json(output, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('split', 'replay'))
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--part-mib', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.part_mib <= 64:
        raise ValueError('Use parts between 1 and 64 MiB.')
    if args.operation == 'split':
        split(args.input, args.output, args.part_mib)
    else:
        replay(args.input, args.output)


if __name__ == '__main__':
    main()
