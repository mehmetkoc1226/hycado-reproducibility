import csv
import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

from hycado_start.common import sha256
from hycado_start.data import prepare
from hycado_start.experiment_data import build_research_data, grouped_split, verify_research_data
from hycado_start.pilot import classify_failure
from hycado_start.stage1 import (
    AccumulationWindow, commit_checkpoint, fingerprint, make_bundle, protocol,
    resume_path, training_lock, trim_uncommitted_csv, reconcile_protocol,
    LEGACY_CHECKPOINT_IO_CODE_SHA256, finish_progress,
)
from test_start import make_raw


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_fixed_budget_and_official_val_is_sealed(self):
        p = protocol()
        self.assertEqual(p["epochs"], 80)
        self.assertEqual(p["physical_batch"], 2)
        self.assertFalse(p["old_experiment_artifacts_used"])
        self.assertIn("not used", p["official_val_policy"])
        self.assertIn("NOT official", p["development_metrics"])
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint(p), fingerprint({**p, "epochs": 81}))

    def test_accumulation_matches_per_image_mean_and_flushes_partial_epoch(self):
        # Per-image scalar gradients; micro-batch losses sum over their images.
        samples = [1., 3., 7., -2., 4., 8., 2., 9., 5., -1., 6.]
        window = AccumulationWindow(8)
        gradient, outputs = 0., []
        batches = [samples[i:i+2] for i in range(0, len(samples), 2)]
        for i, batch in enumerate(batches):
            gradient += sum(batch)
            divisor = window.add_batch(len(batch), last=i == len(batches)-1)
            if divisor is not None:
                outputs.append(gradient/divisor)
                gradient = 0.
        self.assertEqual(outputs, [sum(samples[:8])/8, sum(samples[8:])/3])
        self.assertEqual(window.samples, 0)

    def test_reported_opencv_failure_is_host_memory(self):
        message = "OpenCV(4.11.0) error: (-4:Insufficient memory) Failed to allocate 9000000 bytes in function 'cv::OutOfMemoryError'"
        self.assertEqual(classify_failure(message), "HOST_OUT_OF_MEMORY")
        self.assertEqual(classify_failure("CUDA out of memory"), "OUT_OF_MEMORY")


class SplitTests(unittest.TestCase):
    def test_groups_and_identical_contents_never_cross_development_boundary(self):
        records = []
        for group in range(20):
            for frame in range(2):
                records.append({"image": f"images/train/{group:07d}_{frame:05d}_d_{group:07d}.jpg",
                                "image_sha256": f"hash-{group}-{frame}"})
        records[-1]["image_sha256"] = records[0]["image_sha256"]
        train, development = grouped_split(records)
        self.assertEqual(len(train)+len(development), len(records))
        self.assertTrue(development)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in development})
        self.assertFalse({r["image_sha256"] for r in train} & {r["image_sha256"] for r in development})
        other_train, other_dev = grouped_split(list(reversed(records)))
        self.assertEqual(train, other_train)
        self.assertEqual(development, other_dev)

    def test_all_train_images_including_empty_targets_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, prepared, output = root/"raw", root/"prepared", root/"research"
            make_raw(raw)
            empty = raw/"VisDrone2019-DET-train/annotations/train_000.txt"
            empty.write_text("0,0,10,10,0,0,0,0\n")
            prepare(raw, prepared, allow_small=True)
            official_val = prepared/"labels/val/val_000.txt"
            original_hash = sha256(official_val)
            build_research_data(prepared, output, allow_small=True)
            manifest = verify_research_data(output)
            self.assertEqual(len(manifest["records"]), 40)
            self.assertTrue(all(r["source_split"] == "train" for r in manifest["records"]))
            empty_record = next(r for r in manifest["records"] if r["source_image"].endswith("train_000.png"))
            self.assertEqual(empty_record["gt_count"], 0)
            self.assertEqual(sha256(official_val), original_hash)
            self.assertFalse(manifest["official_val_used_for_training_or_selection"])
            changed = output/"labels"/empty_record["split"]/"train_000.txt"
            changed.write_text("0 .5 .5 .1 .1\n")
            with self.assertRaisesRegex(ValueError, "degismis"):
                verify_research_data(output)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_sync_uses_writable_descriptor_as_required_on_windows(self):
        real_fsync = os.fsync
        descriptors = []
        def writable_fsync(fd):
            # POSIX also rejects write() on a read-only descriptor: reproduce
            # Windows _commit's writable-handle requirement without Windows.
            os.write(fd, b"")
            real_fsync(fd)
            descriptors.append(fd)
        with tempfile.TemporaryDirectory() as tmp, patch("hycado_start.stage1.os.fsync", side_effect=writable_fsync):
            root = Path(tmp)
            commit_checkpoint(root, 1, lambda p: p.write_bytes(b"epoch1"), "protocol")
            self.assertEqual(len(descriptors), 1)
            self.assertEqual(resume_path(root, "protocol").read_bytes(), b"epoch1")

    def test_interrupted_serialization_preserves_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit_checkpoint(root, 1, lambda p: p.write_bytes(b"epoch1"), "protocol")
            index_before = (root/"checkpoint_index.json").read_bytes()
            def broken_writer(path):
                path.write_bytes(b"partial epoch2")
                raise OSError("disk failure")
            with self.assertRaises(OSError):
                commit_checkpoint(root, 2, broken_writer, "protocol")
            self.assertEqual((root/"checkpoint_index.json").read_bytes(), index_before)
            self.assertEqual(resume_path(root, "protocol").read_bytes(), b"epoch1")
            self.assertFalse(list((root/"weights").glob("*.tmp")))

    def test_checkpoint_hash_protocol_and_epoch_overwrite_are_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = commit_checkpoint(root, 1, lambda p: p.write_bytes(b"state"), "protocol")
            with self.assertRaisesRegex(ValueError, "protokol"):
                resume_path(root, "other")
            with self.assertRaises(ValueError):
                commit_checkpoint(root, 1, lambda p: p.write_bytes(b"new"), "protocol")
            (root/item["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                resume_path(root, "protocol")

    def test_retention_keeps_latest_previous_and_tenth_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for epoch in range(1, 14):
                commit_checkpoint(root, epoch, lambda p: p.write_bytes(b"state"), "protocol")
            self.assertEqual({p.name for p in (root/"weights").iterdir()},
                             {"epoch_0010.pt", "epoch_0012.pt", "epoch_0013.pt"})

    def test_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"training.lock"
            with training_lock(path):
                with self.assertRaises(RuntimeError):
                    with training_lock(path):
                        pass
            with training_lock(path):
                pass

    def test_uncommitted_csv_rows_are_not_duplicated_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"results.csv"
            path.write_text("epoch,loss\n1,2\n2,1\n3,.5\n")
            trim_uncommitted_csv(path, 2)
            with path.open(newline="") as stream:
                self.assertEqual(list(csv.reader(stream)), [["epoch", "loss"], ["1", "2"], ["2", "1"]])

    def test_return_bundle_excludes_checkpoints_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"result.json").write_text('{"status":"RUNNING"}')
            (root/"weights").mkdir()
            (root/"weights/epoch_0001.pt").write_bytes(b"weights")
            (root/"console.log").write_text("log")
            import zipfile
            with zipfile.ZipFile(make_bundle(root)) as z:
                self.assertEqual(set(z.namelist()), {"result.json", "console_tail.log"})


class CheckpointFixMigrationTests(unittest.TestCase):
    def make_old_protocol(self, root):
        old = {**protocol(), "dataset_manifest_sha256": "original-data",
               "code_hashes": {"stage1.py": LEGACY_CHECKPOINT_IO_CODE_SHA256, "data.py": "unchanged"}}
        (root/"protocol.json").write_text(json.dumps(old), encoding="utf-8")
        return old

    def test_reported_first_checkpoint_failure_is_migrated_without_changing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self.make_old_protocol(root)
            (root/"training").mkdir()
            (root/"training/results.csv").write_text("epoch,time\n1,613.551\n")
            (root/"result.json").write_text('{"status":"ERROR","error":"Bad file descriptor"}')
            new = {**old, "code_hashes": {**old["code_hashes"], "stage1.py": "fixed-code"}}
            reconcile_protocol(root, new)
            self.assertEqual(json.loads((root/"checkpoint_io_fix_backup/protocol.json").read_text()), old)
            self.assertEqual(json.loads((root/"protocol.json").read_text()), old)
            self.assertEqual((root/"checkpoint_io_fix_backup/training_results.csv").read_text(), "epoch,time\n1,613.551\n")
            migration = json.loads((root/"checkpoint_io_fix_migration.json").read_text())
            self.assertTrue(migration["training_settings_and_data_unchanged"])
            self.assertEqual(migration["restart_epoch"], 1)

    def test_migration_rejects_changed_settings_data_and_existing_checkpoints(self):
        for condition in ("settings", "data", "other_code", "checkpoint", "weights"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old = self.make_old_protocol(root)
                new = {**old, "code_hashes": {**old["code_hashes"], "stage1.py": "fixed-code"}}
                if condition == "settings": new["epochs"] = 79
                if condition == "data": new["dataset_manifest_sha256"] = "different-data"
                if condition == "other_code": new["code_hashes"]["data.py"] = "changed"
                if condition == "checkpoint": (root/"checkpoint_index.json").write_text("{}")
                if condition == "weights":
                    (root/"weights").mkdir()
                    (root/"weights/epoch_0001.pt").write_bytes(b"saved")
                with self.assertRaises(ValueError): reconcile_protocol(root, new)
                self.assertFalse((root/"checkpoint_io_fix_migration.json").exists())

    def test_error_replaces_stale_running_status_and_reports_zero_committed_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"progress.json").write_text('{"status":"RUNNING","epoch":1,"batch":2750}')
            self.assertEqual(finish_progress(root, {"status": "ERROR", "error": "Bad file descriptor"}), 0)
            progress = json.loads((root/"progress.json").read_text())
            self.assertEqual(progress["status"], "ERROR")
            self.assertEqual(progress["last_committed_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
