"""Check all five required stage-26 archives before a CPU prediction replay."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    expected = json.loads((ROOT / 'manifests/global_fusion_inputs.json').read_text())['files']
    for name, sha in expected.items():
        path = Path(folder) / name
        if not path.is_file():
            raise ValueError('Missing required input: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != sha:
            raise ValueError('Input identity differs: ' + name)
        print('VERIFIED:', name, flush=True)
    print('PASSED: all five stage-26 input identities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    try:
        verify(parser.parse_args().inputs)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')
