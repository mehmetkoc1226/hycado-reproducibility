import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('verify_current_publication',Path(__file__).resolve().parents[1]/'tools/verify_v16.py')
v=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v)

class CurrentIdentityTests(unittest.TestCase):
    def fixture(self,root):
        (root/'manifests').mkdir();(root/'docs').mkdir()
        files={'README.md':b'current readme','docs/RELEASE_STATUS.md':b'current status','old.py':b'original code'}
        for name,data in files.items():(root/name).write_bytes(data)
        old={'base_blob_sha':{'README.md':v.gitsha(b'old readme'),'docs/RELEASE_STATUS.md':v.gitsha(b'old status'),'old.py':v.gitsha(files['old.py'])},
             'replaced_documentation_paths':['README.md','docs/RELEASE_STATUS.md']}
        current={name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha':v.gitsha(data)} for name,data in files.items() if name!='old.py'}
        (root/'manifests/v16_base_blobs.json').write_text(json.dumps(old));(root/'manifests/v16_extension_files.json').write_text(json.dumps(current))

    def test_current_docs_and_old_code_have_distinct_verified_identities(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            self.assertEqual(v.verify_files(root),{'historical_blobs_preserved':1,'current_files_verified':2,'replaced_docs_verified':2})
            (root/'README.md').write_text('tampered current doc')
            with self.assertRaises(ValueError):v.verify_files(root)

    def test_old_code_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);(root/'old.py').write_text('changed code')
            with self.assertRaises(ValueError):v.verify_files(root)

if __name__=='__main__':unittest.main()
