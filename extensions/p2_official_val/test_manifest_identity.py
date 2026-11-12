"""Small, CPU-only regression check for the corrected official manifest guard."""
import copy
import unittest
from unittest.mock import patch

import p2_official_val as official


class ManifestIdentityTest(unittest.TestCase):
    def setUp(self):
        self.pre={'split_identity':{'official_val':{
            'source_manifest_sha256':official.OLD_BASELINE_PROTOCOL_FILE_SHA}}}
        self.old={'prepared_manifest_sha256':official.PREPARED_MANIFEST_SHA}

    def check(self, pre=None, old=None, actual_manifest=None, actual_protocol=None):
        hashes={
            'old_protocol.json':actual_protocol or official.OLD_BASELINE_PROTOCOL_FILE_SHA,
            'manifest.json':actual_manifest or official.PREPARED_MANIFEST_SHA,
        }
        with patch.object(official.d,'digest',side_effect=lambda p: hashes[str(p)]), \
             patch.object(official.d,'read',return_value=old or self.old):
            return official.verify_manifest_identity(pre or self.pre,'manifest.json','old_protocol.json')

    def test_historical_field_is_protocol_file_hash(self):
        self.assertEqual(self.check(),self.old)
        self.assertNotEqual(official.OLD_BASELINE_PROTOCOL_FILE_SHA,official.PREPARED_MANIFEST_SHA)

    def test_changed_preflight_field_rejected(self):
        pre=copy.deepcopy(self.pre)
        pre['split_identity']['official_val']['source_manifest_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'baseline protocol'):
            self.check(pre=pre)

    def test_changed_old_protocol_file_rejected(self):
        with self.assertRaisesRegex(ValueError,'baseline protocol'):
            self.check(actual_protocol='0'*64)

    def test_changed_old_protocol_manifest_binding_rejected(self):
        with self.assertRaisesRegex(ValueError,'data audit'):
            self.check(old={'prepared_manifest_sha256':'0'*64})

    def test_changed_current_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError,'Official original manifest differs'):
            self.check(actual_manifest='0'*64)


if __name__=='__main__':
    unittest.main()
