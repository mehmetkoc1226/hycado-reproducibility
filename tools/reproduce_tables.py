"""Replay presentation assets from frozen metrics; does not recalculate AP."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    source = ROOT / 'paper_assets'
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output cannot replace or contain the frozen paper assets')
    output.mkdir(parents=True, exist_ok=True)
    for name in ['evidence', 'scripts', 'tables']:
        shutil.copytree(source / name, output / name)
    (output / 'figures').mkdir()
    subprocess.run([sys.executable, str(output / 'scripts/generate_results.py')], check=True)
    expected = json.loads((ROOT / 'manifests/expected_tables_sha256.json').read_text())
    actual = {name: hashlib.sha256((output / 'tables' / name).read_bytes()).hexdigest()
              for name in expected}
    if actual != expected:
        raise ValueError('Regenerated tables differ from the retained paper tables')
    report = {'status': 'PASSED', 'tables_verified': len(expected), 'sha256': actual,
              'scope': 'table/plot regeneration from recorded metrics; no prediction replay, training or timing'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/table_replay')
    try:
        run(parser.parse_args().output)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')
