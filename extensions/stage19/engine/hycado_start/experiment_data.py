"""Immutable, audited TRAIN/development split for the first research training."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .common import sha256, utc_now, write_json
from .data import EXPECTED, NAMES, _link_or_copy, _write_dataset_yaml, parse_annotation
from .stability import inspect_labels


def filename_group(image):
    stem = PurePosixPath(image).stem
    # This is a filename grouping rule, not a verified scene/video identity.
    return stem.split("_", 1)[0] if re.fullmatch(r"\d{7}_\d{5}_d_\d{7}", stem) else stem


def grouped_split(records, seed=20261022, fraction=0.10):
    if not 0 < fraction < 0.5:
        raise ValueError("Gelistirme orani 0 ile 0.5 arasinda olmali")
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    by_hash = {}
    for record in sorted(records, key=lambda r: r["image"]):
        group = filename_group(record["image"])
        find(group)
        old = by_hash.setdefault(record["image_sha256"], group)
        a, b = find(group), find(old)
        parents[max(a, b)] = min(a, b)
    groups = {}
    for record in records:
        groups.setdefault(find(filename_group(record["image"])), []).append(record)
    if len(groups) < 2:
        raise ValueError("Gelistirme ayrimi icin en az iki bagimsiz dosya grubu gerekli")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    target = max(1, round(len(records) * fraction))
    chosen, count = set(), 0
    for key in ordered:
        new_count = count + len(groups[key])
        if new_count < len(records) and abs(new_count-target) < abs(count-target):
            chosen.add(key)
            count = new_count
    if not chosen:
        chosen.add(min(groups, key=lambda key: (len(groups[key]), key)))
    train, development = [], []
    for key in sorted(groups):
        for record in sorted(groups[key], key=lambda r: r["image"]):
            (development if key in chosen else train).append({**record, "group": key})
    if not train or not development:
        raise ValueError("Train/gelistirme ayrimi bos olamaz")
    return train, development


def build_research_data(prepared, output, *, allow_small=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Deney verisinin uzerine yazilmadi: {output}")
    source_manifest = prepared / "manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("purpose") != "fresh_original_visdrone_conversion":
        raise ValueError("Temiz veri hazirlama manifesti gerekli")
    counts = Counter(r["split"] for r in source["records"])
    if not allow_small and (dict(counts) != EXPECTED or source.get("is_small_fixture")):
        raise ValueError(f"Resmi train/val sayilari gerekli: {dict(counts)}")
    originals = [r for r in source["records"] if r["split"] == "train"]
    if len({r["image"] for r in originals}) != len(originals):
        raise ValueError("Kaynak manifestte yinelenen goruntu yolu")
    records, clean_labels, issues = [], {}, []
    for index, record in enumerate(originals):
        relative = PurePosixPath(record["image"])
        if (len(relative.parts) != 3 or relative.parts[:2] != ("images", "train") or
                ".." in relative.parts or "\\" in str(relative)):
            raise ValueError("Kaynak train goruntu yolu gecersiz")
        image = prepared / record["image"]
        annotation = prepared / "visdrone_annotations/train" / (relative.stem + ".txt")
        label = prepared / "labels/train" / (relative.stem + ".txt")
        if sha256(image) != record["image_sha256"] or sha256(annotation) != record["annotation_sha256"]:
            raise ValueError(f"Kaynak veri hash'i degismis: {relative.name}")
        converted = parse_annotation(annotation.read_text(encoding="utf-8-sig"),
                                     record["width"], record["height"], source=str(annotation))
        expected = "".join(str(row[0]) + " " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                           for row in converted["labels"])
        if label.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Hazir etiket ham donusumle uyusmuyor: {relative.name}")
        clean, duplicates, gt_count = inspect_labels(expected, str(label))
        class_counts = Counter(int(line.split()[0]) for line in clean.splitlines())
        records.append({**record, "gt_count": gt_count, "class_counts": dict(class_counts),
                        "source_label_sha256": sha256(label)})
        clean_labels[record["image"]] = clean
        if duplicates or converted["issues"]:
            issues.append({"source_image": record["image"], "duplicates": duplicates,
                           "invalid_geometry": converted["issues"]})
        if index % 500 == 0:
            print(f"Gercek deney verisi dogrulaniyor: {index+1}/{len(originals)}", flush=True)
    train, development = grouped_split(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + "_building_", dir=output.parent))
    manifest_records = []
    try:
        for split, selected in (("train", train), ("val", development)):
            for record in selected:
                source_image = prepared / record["image"]
                relative = f"images/{split}/{source_image.name}"
                _link_or_copy(source_image, temporary / relative)
                label = temporary / "labels" / split / (source_image.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(clean_labels[record["image"]], encoding="utf-8")
                annotation = temporary / "annotations" / split / (source_image.stem + ".txt")
                annotation.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prepared / "visdrone_annotations/train" / annotation.name, annotation)
                manifest_records.append({"source_image": record["image"], "source_split": "train",
                                         "split": split, "group": record["group"], "image": relative,
                                         "image_sha256": record["image_sha256"],
                                         "label_sha256": sha256(label),
                                         "annotation_sha256": record["annotation_sha256"],
                                         "gt_count": record["gt_count"], "class_counts": record["class_counts"]})
        summary = {split: {"images": len(selected), "groups": len({r["group"] for r in selected}),
                           "class_counts": {name: sum(r["class_counts"].get(cls, 0) for r in selected)
                                            for cls, name in enumerate(NAMES)}}
                   for split, selected in (("train", train), ("development", development))}
        write_json(temporary / "manifest.json", {"purpose": "stage1_research_development_training_v1",
                   "created_utc": utc_now(), "records": manifest_records, "summary": summary,
                   "source_manifest_sha256": sha256(source_manifest), "split_seed": 20261022,
                   "development_target_fraction": 0.10,
                   "group_policy": "filename first token for standard VisDrone names; exact duplicate image hashes union groups; not verified scene independence",
                   "official_val_used_for_training_or_selection": False,
                   "empty_label_images_kept": True})
        write_json(temporary / "label_audit.json", {"issues": issues,
                   "duplicate_rows_removed": sum(len(i["duplicates"]) for i in issues),
                   "invalid_geometry_rows_omitted": sum(len(i["invalid_geometry"]) for i in issues),
                   "raw_annotation_policy": "byte-exact copies; changes only to derived training/development labels"})
        _write_dataset_yaml(temporary / "dataset.yaml", output)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("Sabit veri ayrimi:", json.dumps(summary), flush=True)
    return output / "dataset.yaml"


def verify_research_data(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = set()
    for index, record in enumerate(manifest["records"]):
        relative = PurePosixPath(record["image"])
        if len(relative.parts) != 3 or relative.parts[:2] not in {("images", "train"), ("images", "val")} or ".." in relative.parts:
            raise ValueError("Deney manifest yolu gecersiz")
        split = relative.parts[1]
        paths = [(root / record["image"], record["image_sha256"]),
                 (root / "labels" / split / (relative.stem + ".txt"), record["label_sha256"]),
                 (root / "annotations" / split / (relative.stem + ".txt"), record["annotation_sha256"])]
        for path, expected_hash in paths:
            if sha256(path) != expected_hash:
                raise ValueError(f"Deney verisi degismis: {path}")
        expected.add(record["image"])
        if index % 1000 == 0:
            print(f"Deney hash kontrolu: {index+1}/{len(manifest['records'])}", flush=True)
    actual = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Deney goruntu listesine dosya eklenmis/silinmis")
    return manifest
