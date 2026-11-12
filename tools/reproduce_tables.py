"""Regenerate v12 or historical presentation assets from recorded metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def run(output, revision='v12'):
    if revision not in ('v12', 'legacy'):
        raise ValueError('Unknown revision: ' + revision)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    assets = ROOT / 'paper_assets'
    if output == assets or assets in output.parents or output in assets.parents:
        raise ValueError('Output cannot replace or contain the frozen paper assets')
    source = assets / 'v12' if revision == 'v12' else assets
    output.mkdir(parents=True, exist_ok=True)
    for name in ['evidence', 'scripts', 'tables']:
        shutil.copytree(source / name, output / name)
    (output / 'figures').mkdir()
    subprocess.run([sys.executable, str(output / 'scripts/generate_results.py')], check=True)
    manifest = 'expected_tables_v12_sha256.json' if revision == 'v12' else 'expected_tables_sha256.json'
    expected = json.loads((ROOT / 'manifests' / manifest).read_text())
    actual = {name: hashlib.sha256((output / 'tables' / name).read_bytes()).hexdigest()
              for name in expected}
    if actual != expected:
        raise ValueError('Regenerated tables differ from the retained paper tables')
    report = {'status': 'PASSED', 'revision': revision, 'latex_files_verified': len(expected),
              'sha256': actual, 'scope': 'recorded-result regeneration; no prediction replay or training'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', choices=['v12', 'legacy'], default='v12')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/table_replay')
    args = parser.parse_args()
    try:
        run(args.output, args.revision)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')
