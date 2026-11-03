import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from hycado_start.common import sha256
from hycado_start.data import prepare
from hycado_start.stability import (
    build_stability_data, bundle_reports, collect_child_result, inspect_labels,
    run, stability_options,
)
from test_start import make_raw


class LabelAuditTests(unittest.TestCase):
    def test_only_exact_numeric_duplicates_are_removed(self):
        text = ("0 0.5 0.5 0.2 0.2\n0 .50 .50 .20 .20\n"
                "1 0.5 0.5 0.2 0.2\n0 0.50001 0.5 0.2 0.2\n")
        clean, duplicates, count = inspect_labels(text, "fixture.txt")
        self.assertEqual(count, 3)
        self.assertEqual(len(clean.splitlines()), 3)
        self.assertEqual(duplicates, [{"line_number": 2, "first_line_number": 1,
                                      "values": [0, .5, .5, .2, .2]}])

    def test_invalid_derived_geometry_and_classes_fail(self):
        for text in ["0 .5 .5 0 .2", "0 .95 .5 .2 .2", "10 .5 .5 .2 .2",
                     "0 nan .5 .2 .2", "0 .5 .5 .2", "bad .5 .5 .2 .2"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                inspect_labels(text, "fixture.txt")

    def test_effective_batch_is_divisible_and_recipe_is_explicit(self):
        opts = stability_options({"epochs": 3, "batch": 2, "effective_batch": 8})
        self.assertEqual(opts["nbs"], 8)
        self.assertEqual(opts["warmup_epochs"], 0)
        self.assertGreater(opts["mosaic"], 0)
        self.assertFalse(opts["multi_scale"])
        with self.assertRaises(ValueError):
            stability_options({"epochs": 3, "batch": 3, "effective_batch": 8})


class StabilityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, self.prepared = self.root / "raw", self.root / "prepared"
        make_raw(self.raw)
        prepare(self.raw, self.prepared, allow_small=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pilot_deduplicates_copies_and_preserves_originals(self):
        source_hashes = {p: sha256(p) for p in (self.prepared / "labels").rglob("*.txt")}
        dataset = build_stability_data(self.prepared, self.root / "pilot", 24, 8)
        root = dataset.parent
        manifest = json.loads((root / "manifest.json").read_text())
        records = manifest["records"]
        self.assertEqual(len(records), 32)
        self.assertTrue(all(r["split"] == "train" for r in records))
        self.assertEqual(sum(r["pilot_split"] == "train" for r in records), 24)
        self.assertEqual(len({r["image_sha256"] for r in records}), 32)
        audit = json.loads((root / "label_audit.json").read_text())
        self.assertEqual(audit["audited_images"], 42)
        self.assertGreater(audit["duplicate_rows"], 0)
        for record in records:
            label = root / "labels" / record["pilot_split"] / (Path(record["image"]).stem + ".txt")
            self.assertEqual(len(label.read_text().splitlines()), 1)
            self.assertEqual(sha256(label), record["pilot_label_sha256"])
        self.assertTrue(all(sha256(p) == value for p, value in source_hashes.items()))

    def test_tampered_label_with_same_row_count_is_rejected(self):
        label = self.prepared / "labels/train/train_000.txt"
        label.write_text("0 .5 .5 .1 .1\n")
        with self.assertRaisesRegex(ValueError, "uyusmuyor"):
            build_stability_data(self.prepared, self.root / "pilot", 24, 8)
        self.assertFalse((self.root / "pilot").exists())

    def test_changed_source_annotation_is_rejected(self):
        raw = self.prepared / "visdrone_annotations/train/train_000.txt"
        raw.write_text("0,0,10,10,1,1,0,0\n")
        with self.assertRaisesRegex(ValueError, "annotation degismis"):
            build_stability_data(self.prepared, self.root / "pilot", 24, 8)

    def test_changed_selected_image_cleans_partial_output(self):
        # All 40 source images are changed; whichever is chosen must fail.
        for image in (self.prepared / "images/train").iterdir():
            image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Goruntu degismis"):
            build_stability_data(self.prepared, self.root / "pilot", 24, 8)
        self.assertFalse((self.root / "pilot").exists())


class StabilityReportTests(unittest.TestCase):
    def test_crashed_process_cannot_leave_a_passing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result.json"
            result.write_text(json.dumps({"status": "PASS_STABILITY_ONLY"}))
            self.assertEqual(collect_child_result(root, 1)["status"], "PROCESS_FAILED")
            result.write_text(json.dumps({"status": "STARTED", "completed_train_batches": 25}))
            report = collect_child_result(root, timed_out=True)
            self.assertEqual(report["status"], "TIMEOUT")
            self.assertEqual(report["completed_train_batches"], 25)

    def test_manager_failure_still_returns_compact_diagnostic_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, success = run(root / "missing", root / "reports", n_train=24, n_val=8)
            self.assertFalse(success)
            with zipfile.ZipFile(bundle) as z:
                summary = json.loads(z.read("summary.json"))
                self.assertIsNotNone(summary["manager_error"])
                self.assertIn(b"Traceback", z.read("console.log"))
                self.assertFalse(summary["actual_stage1_crops_tested"])

    def test_bundle_excludes_images_weights_and_arbitrary_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stability_fixture"
            root.mkdir()
            for name in ["summary.json", "console.log", "image.jpg", "weights.pt", "dataset.yaml", "data.cache"]:
                (root / name).write_text("test")
            bundle = bundle_reports(root)
            with zipfile.ZipFile(bundle) as z:
                self.assertEqual(set(z.namelist()), {"summary.json", "console.log"})


if __name__ == "__main__":
    unittest.main()
