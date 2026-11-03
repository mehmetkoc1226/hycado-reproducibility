"""Check the package's file identities, source provenance and exact split roles."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    entries = json.loads((ROOT / 'manifests/release_files_sha256.json').read_text())
    for name, expected in entries.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise ValueError('Missing/invalid package path: ' + name)
        if digest(path) != expected:
            raise ValueError('File identity differs: ' + name)
    inventory = json.loads((ROOT / 'splits/image_identities.json').read_text())
    records = inventory['records']
    if len(records) != 7019 or len({r['source_image'] for r in records}) != 7019:
        raise ValueError('Source inventory must contain 7019 unique filenames')
    sets = {}
    for role, count in {'training': 5824, 'development': 647, 'official_val': 548}.items():
        names = (ROOT / 'splits' / (role + '.txt')).read_text().splitlines()
        expected = sorted(r['source_image'] for r in records if r['role'] == role)
        if len(names) != count or names != expected:
            raise ValueError('Split membership differs: ' + role)
        sets[role] = set(names)
    if any(sets[a] & sets[b] for a, b in [('training', 'development'), ('training', 'official_val'), ('development', 'official_val')]):
        raise ValueError('Filename roles overlap')
    groups = {}
    for r in records:
        groups.setdefault(r['rgb_sha256'], set()).add(r['role'])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError('Recorded exact RGB identity crosses roles')
    print(json.dumps({'status': 'PASSED', 'files_verified': len(entries), 'source_images': len(records),
                      'roles': {k: len(v) for k, v in sets.items()},
                      'scope': 'packaged identities and recorded roles; original images not reopened'}, indent=2))


if __name__ == '__main__':
    main()
