"""Check refusal paths that protect source files and prevent false input validation."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tables = load('reproduce_tables')
inputs = load('verify_inputs')


class PackagingTests(unittest.TestCase):
    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'keep.txt'; p.write_text('retained')
            with self.assertRaises(ValueError): tables.run(folder)
            self.assertEqual(p.read_text(), 'retained')

    def test_frozen_asset_directory_cannot_be_target(self):
        with self.assertRaises(ValueError): tables.run(ROOT / 'paper_assets')

    def test_inputs_validate_bytes_not_just_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'manifests').mkdir(); data = root / 'inputs'; data.mkdir()
            (data / 'a.zip').write_bytes(b'retained-predictions')
            (root / 'manifests/global_fusion_inputs.json').write_text(json.dumps({'files': {
                'a.zip': hashlib.sha256(b'retained-predictions').hexdigest()}}))
            with patch.object(inputs, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                inputs.verify(data)
                (data / 'a.zip').write_bytes(b'changed-predictions')
                with self.assertRaises(ValueError): inputs.verify(data)
                (data / 'a.zip').unlink()
                with self.assertRaises(ValueError): inputs.verify(data)


if __name__ == '__main__':
    unittest.main()
