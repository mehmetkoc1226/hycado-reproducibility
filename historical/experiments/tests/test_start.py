import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hycado_start.common import sha256, write_json
from hycado_start.data import parse_annotation, prepare, make_capacity_subset, choose_capacity_records
from hycado_start.pilot import classify_failure
from hycado_start.__main__ import prepare_logged


class ConversionTests(unittest.TestCase):
    def test_class_mapping_and_ignore_preservation(self):
        text = "0,0,10,20,1,1,0,0\n10,0,10,20,1,10,0,0\n0,0,50,50,0,0,0,0\n"
        text += "0,0,10,10,0,3,0,0\n0,0,10,10,1,11,0,0\n"
        result = parse_annotation(text, 100, 100)
        self.assertEqual([r[0] for r in result["labels"]], [0, 9])
        self.assertEqual(len(result["ignore_boxes_xyxy"]), 2)
        self.assertEqual(len(result["other_boxes_xyxy"]), 1)
        self.assertEqual(len(result["raw_rows"]), 5)

    def test_clipping_uses_actual_image_dimensions(self):
        result = parse_annotation("-5,-5,15,15,1,4,0,0,\n", 100, 50)
        self.assertEqual(result["labels"], [[3, .05, .1, .1, .2]])

    def test_malformed_labels_fail(self):
        for text in ["1,2,3", "nan,0,10,10,1,1,0,0", "0,0,10,10,1,12,0,0",
                     "bad,0,10,10,1,1,0,0"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_annotation(text, 100, 100)

    def test_invalid_geometry_is_audited_without_inventing_boxes(self):
        text = "0,0,8,8,1,4,0,0\n0,0,0,10,1,1,0,0\n0,0,5,-1,0,0,0,0\n200,0,10,10,1,11,0,0\n"
        result = parse_annotation(text, 100, 100, source="sample.txt")
        self.assertEqual(len(result["labels"]), 1)
        self.assertEqual(len(result["raw_rows"]), 4)
        self.assertEqual([i["line_number"] for i in result["issues"]], [2, 3, 4])
        self.assertEqual([i["is_target"] for i in result["issues"]], [True, False, False])
        self.assertEqual(result["issues"][0]["row"], text.splitlines()[1].split(","))
        self.assertEqual(result["issues"][2]["reason"], "no_area_inside_image")

    def test_fatal_annotation_error_includes_source_line_and_content(self):
        with self.assertRaisesRegex(ValueError, r"sample.txt.*satir 2.*bad,0"):
            parse_annotation("\nbad,0,10,10,1,1,0,0", 100, 100, source="sample.txt")


def make_raw(root, train_count=40, val_count=2):
    for split, count in (("train", train_count), ("val", val_count)):
        folder = root / f"VisDrone2019-DET-{split}"
        (folder / "images").mkdir(parents=True)
        (folder / "annotations").mkdir()
        for index in range(count):
            stem = f"{split}_{index:03d}"
            color = (index, 10 if split == "train" else 220, 30)
            Image.new("RGB", (64, 48), color).save(folder / "images" / (stem + ".png"))
            count_gt = 1 + index % 12
            text = "0,0,8,8,1,1,0,0\n" * count_gt + "32,24,10,10,0,0,0,0\n"
            (folder / "annotations" / (stem + ".txt")).write_text(text, encoding="utf-8")


class DatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        make_raw(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversion_subset_and_raw_annotations_remain_traceable(self):
        out = self.root / "prepared"
        prepare(self.raw, out, allow_small=True)
        self.assertEqual(len(list((out / "images/train").glob("*.png"))), 40)
        self.assertTrue((out / "visdrone_annotations/train/train_000.txt").exists())
        dataset_yaml = make_capacity_subset(out, self.root / "capacity")
        manifest = json.loads(dataset_yaml.with_name("manifest.json").read_text())
        records = manifest["records"]
        self.assertEqual(len(records), 32)
        self.assertEqual(len({r["image"] for r in records}), 32)
        self.assertTrue(all(r["split"] == "train" for r in records))
        self.assertEqual(sum(r["capacity_split"] == "train" for r in records), 24)
        self.assertEqual(sum(r["capacity_split"] == "val" for r in records), 8)

    def test_missing_label_fails_before_writing(self):
        (self.raw / "VisDrone2019-DET-train/annotations/train_000.txt").unlink()
        out = self.root / "prepared"
        with self.assertRaises(FileNotFoundError):
            prepare(self.raw, out, allow_small=True)
        self.assertFalse(out.exists())

    def test_cross_split_identical_images_are_rejected(self):
        train = self.raw / "VisDrone2019-DET-train/images/train_000.png"
        val = self.raw / "VisDrone2019-DET-val/images/val_000.png"
        val.write_bytes(train.read_bytes())
        out = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "birebir"):
            prepare(self.raw, out, allow_small=True)
        self.assertFalse(out.exists())

    def test_existing_output_is_not_overwritten(self):
        out = self.root / "prepared"
        out.mkdir()
        marker = out / "marker.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare(self.raw, out, allow_small=True)
        self.assertEqual(marker.read_text(), "keep")

    def test_modified_prepared_image_is_rejected(self):
        out = self.root / "prepared"
        prepare(self.raw, out, allow_small=True)
        manifest = json.loads((out / "manifest.json").read_text())
        chosen = choose_capacity_records(manifest["records"])[0][0]
        image = out / chosen["image"]
        image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "degismis"):
            make_capacity_subset(out, self.root / "capacity")

    def test_real_split_counts_are_required_by_default(self):
        with self.assertRaisesRegex(ValueError, "6471"):
            prepare(self.raw, self.root / "prepared")

    def test_empty_target_images_remain_in_prepared_dataset(self):
        annotation = self.raw / "VisDrone2019-DET-train/annotations/train_000.txt"
        annotation.write_text("0,0,10,10,0,0,0,0\n")
        out = self.root / "prepared"
        prepare(self.raw, out, allow_small=True)
        self.assertTrue((out / "images/train/train_000.png").exists())
        self.assertEqual((out / "labels/train/train_000.txt").read_text(), "")

    def test_invalid_rows_preserve_raw_image_and_other_targets(self):
        annotation = self.raw / "VisDrone2019-DET-train/annotations/train_000.txt"
        original = annotation.read_bytes() + b"0,0,0,10,1,1,0,0\r\n"
        annotation.write_bytes(original)
        out = self.root / "prepared"
        prepare(self.raw, out, allow_small=True)
        self.assertEqual((out / "visdrone_annotations/train/train_000.txt").read_bytes(), original)
        self.assertEqual(annotation.read_bytes(), original)
        self.assertTrue((out / "images/train/train_000.png").exists())
        self.assertEqual(len((out / "labels/train/train_000.txt").read_text().splitlines()), 1)
        report = json.loads((out / "data_quality.json").read_text())
        self.assertEqual(report["invalid_geometry_rows"], 1)
        self.assertEqual(report["invalid_target_rows"], 1)
        self.assertEqual(report["issues"][0]["line_number"], 3)
        self.assertEqual(report["issues"][0]["source"], str(annotation))
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["records"][0]["annotation_sha256"], sha256(annotation))

    def test_failure_has_persistent_log_and_no_stale_quality_report(self):
        reports = self.root / "reports"
        reports.mkdir()
        quality = reports / "02_veri_kalite_raporu.json"
        quality.write_text("old result")
        self.assertEqual(prepare_logged(self.root / "missing", self.root / "prepared", reports), 1)
        log = (reports / "02_veri_hazirlama.log").read_text(encoding="utf-8")
        self.assertIn("Traceback", log)
        self.assertIn("FileNotFoundError", log)
        self.assertFalse(quality.exists())


class ReportingTests(unittest.TestCase):
    def test_memory_and_numerical_failures_are_distinct(self):
        self.assertEqual(classify_failure("CUDA out of memory"), "OUT_OF_MEMORY")
        self.assertEqual(classify_failure("matrix contains invalid numeric entries"), "NUMERICAL_FAILURE")
        self.assertEqual(classify_failure("non-finite training loss"), "NUMERICAL_FAILURE")
        self.assertEqual(classify_failure("network unavailable"), "ERROR")

    def test_json_is_atomic_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"a": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
