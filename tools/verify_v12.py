"""Check v12 tables, exact extension sources and recorded split identities on a CPU."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'paper_assets/v12'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT / 'manifests/expected_tables_v12_sha256.json').read_bytes())
    require(set(manifest) == {p.name for p in (ASSETS / 'tables').glob('*.tex')}, 'Table coverage differs')
    for name, digest in manifest.items():
        require(sha(ASSETS / 'tables' / name) == digest, 'Table bytes differ: ' + name)
    packages = json.loads((ROOT / 'manifests/extensions_sources.json').read_bytes())['packages']
    files = 0
    for package in packages:
        folder = ROOT / package['repository_path']
        for name, digest in package['member_sha256'].items():
            path = (folder / name).resolve()
            require(folder.resolve() in path.parents, 'Invalid source path')
            require(path.is_file() and sha(path) == digest, 'Extension source differs: ' + str(path))
            files += 1
    reports = ASSETS / 'evidence/extensions'
    for stage, folder, bindings in [
        (18, '18_testdev_dogrulama_rapor', {'testdev.py': 'runner_sha256'}),
        (20, '20_testdev1280_rapor', {'test1280.py': 'runner_sha256',
                                    'frozen_support.py': 'support_sha256',
                                    'evidence/stage19_parity.json': 'parity_records_sha256'})]:
        protocol = json.loads((reports / folder / 'protocol.json').read_bytes())
        for name, field in bindings.items():
            require(sha(ROOT / 'extensions' / ('stage' + str(stage)) / name) == protocol[field],
                    'Recorded runner binding differs: ' + name)
    protocol = json.loads((reports / '19_tam1280_gelistirme_rapor/protocol.json').read_bytes())
    for name, digest in protocol['source_sha256'].items():
        require(sha(ROOT / 'extensions/stage19' / name) == digest, 'Stage19 source binding differs: ' + name)
    records = json.loads((reports / '18_testdev_dogrulama_rapor/dataset_manifest.json').read_bytes())['records']
    names = (ROOT / 'splits/test_dev.txt').read_text().splitlines()
    require(names == [r['image'] for r in records] and len(set(names)) == len(names) == 1610,
            'Test-dev membership differs')
    original = json.loads((ROOT / 'splits/image_identities.json').read_bytes())['records']
    require(not set(names) & {r['source_image'] for r in original}, 'Logical paths cross splits')
    original_hashes = {r['sha256'] for r in original}
    original_hashes.discard(None)
    require(bool(original_hashes), 'Original JPEG identity field not found')
    require(not {r['image_sha256'] for r in records} & original_hashes, 'JPEG bytes cross splits')
    for script in ['check_extensions.py', 'check_val_diagnostics.py']:
        subprocess.run([sys.executable, str(ASSETS / 'scripts' / script)], check=True)
    print(json.dumps({'status': 'PASSED', 'latex_files': len(manifest),
                      'extension_source_files': files, 'test_dev_images': len(names),
                      'new_training': False, 'new_inference': False,
                      'scope': 'recorded source/report identities and table consistency'}, indent=2))

if __name__ == '__main__':
    main()
