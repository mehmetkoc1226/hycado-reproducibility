"""Fresh VisDrone annotation conversion and a training-only capacity subset.

This is NOT an official VisDrone evaluator. Raw annotations are preserved so
the later evaluator can implement the official ignore-region protocol.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from .common import sha256, utc_now, write_json

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
EXPECTED = {"train": 6471, "val": 548}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_annotation(text, width, height, source="<annotation>"):
    labels, ignored, others, rows, issues = [], [], [], [], []
    if width <= 0 or height <= 0:
        raise ValueError(f"{source}: goruntu boyutlari pozitif olmali")
    for line_number, row in enumerate(csv.reader(text.splitlines()), 1):
        original_row = list(row)
        context = f"{source} | Annotation satir {line_number} | {','.join(original_row)}"
        row = [field.strip() for field in row]
        while row and row[-1] == "":
            row.pop()
        if not row:
            continue
        if len(row) != 8:
            raise ValueError(f"{context}: 8 alan bekleniyor")
        try:
            values = [float(field) for field in row]
        except ValueError as exc:
            raise ValueError(f"{context}: sayiya donusturulemeyen alan") from exc
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{context}: sonlu olmayan deger")
        x, y, bw, bh, score, cls, trunc, occ = values
        if cls != int(cls) or not 0 <= cls <= 11 or score not in (0, 1):
            raise ValueError(f"{context}: sinif/skor gecersiz")
        rows.append(values)
        box = [max(0., x), max(0., y), min(float(width), x+bw), min(float(height), y+bh)]
        reason = ("nonpositive_box_size" if bw <= 0 or bh <= 0 else
                  "no_area_inside_image" if box[2] <= box[0] or box[3] <= box[1] else None)
        if reason:
            issues.append({"source": str(source), "line_number": line_number,
                           "row": original_row, "values": values, "reason": reason,
                           "is_target": bool(score == 1 and 1 <= cls <= 10),
                           "action": "omit_from_derived_geometry_preserve_raw"})
            continue
        if cls == 0 or score == 0:
            ignored.append(box)
        elif cls == 11:
            others.append(box)
        else:
            x1, y1, x2, y2 = box
            labels.append([int(cls)-1, (x1+x2)/(2*width), (y1+y2)/(2*height),
                           (x2-x1)/width, (y2-y1)/height])
    return {"labels": labels, "ignore_boxes_xyxy": ignored,
            "other_boxes_xyxy": others, "raw_rows": rows, "issues": issues}


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_dataset_yaml(path, dataset_root):
    import yaml
    document = {"path": str(Path(dataset_root).resolve()), "train": "images/train",
                "val": "images/val", "nc": 10, "names": NAMES}
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare(raw_root, output, allow_small=False):
    """Convert original train/val splits. Never overwrite a prior prepared tree."""
    from PIL import Image
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Hedef zaten var; uzerine yazilmadi: {output}")
    sources = {}
    for split in EXPECTED:
        directory = raw_root / f"VisDrone2019-DET-{split}"
        if not (directory / "images").is_dir() or not (directory / "annotations").is_dir():
            raise FileNotFoundError(f"Orijinal images ve annotations klasorleri gerekli: {directory}")
        images = sorted(p for p in (directory / "images").iterdir() if p.suffix.lower() in SUFFIXES)
        if not images or (not allow_small and len(images) != EXPECTED[split]):
            raise ValueError(f"{split}: {len(images)} goruntu bulundu; beklenen {EXPECTED[split]}")
        if len({p.stem.casefold() for p in images}) != len(images):
            raise ValueError(f"{split}: ayni kok adina sahip goruntuler var")
        for image in images:
            if not (directory / "annotations" / (image.stem + ".txt")).is_file():
                raise FileNotFoundError(f"Etiket eksik: {split}/{image.stem}.txt")
        sources[split] = directory, images
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + "_preparing_", dir=output.parent))
    records, seen_hashes, issues = [], {}, []
    try:
        for split, (directory, images) in sources.items():
            for index, source in enumerate(images):
                annotation = directory / "annotations" / (source.stem + ".txt")
                with Image.open(source) as image:
                    width, height = image.size
                    image.verify()
                converted = parse_annotation(annotation.read_text(encoding="utf-8-sig"), width, height,
                                             source=str(annotation))
                issues.extend({**issue, "split": split, "image": source.name,
                               "image_width": width, "image_height": height}
                              for issue in converted["issues"])
                image_hash = sha256(source)
                if image_hash in seen_hashes and seen_hashes[image_hash] != split:
                    raise ValueError("Train/val arasinda birebir ayni goruntu bulundu")
                seen_hashes[image_hash] = split
                relative_image = f"images/{split}/{source.name}"
                _link_or_copy(source, temporary / relative_image)
                label = temporary / "labels" / split / (source.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("".join(str(row[0]) + " " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                                         for row in converted["labels"]), encoding="utf-8")
                raw_copy = temporary / "visdrone_annotations" / split / annotation.name
                raw_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(annotation, raw_copy)
                records.append({"split": split, "image": relative_image, "width": width,
                                "height": height, "image_sha256": image_hash,
                                "annotation_sha256": sha256(annotation),
                                "gt_count": len(converted["labels"]),
                                "invalid_geometry_count": len(converted["issues"]),
                                "ignore_boxes_xyxy": converted["ignore_boxes_xyxy"],
                                "other_boxes_xyxy": converted["other_boxes_xyxy"]})
                if index % 500 == 0:
                    print(f"Veri hazirlaniyor: {split} {index+1}/{len(images)}", flush=True)
        quality = {"created_utc": utc_now(), "purpose": "capacity_preparation_data_quality",
                   "policy": "invalid geometry omitted only from derived labels/boxes; raw annotations unchanged; review before scientific experiments",
                   "invalid_geometry_rows": len(issues),
                   "invalid_target_rows": sum(i["is_target"] for i in issues),
                   "affected_images": len({(i["split"], i["image"]) for i in issues}),
                   "by_split": {split: {"images": sum(r["split"] == split for r in records),
                                          "invalid_geometry_rows": sum(i["split"] == split for i in issues),
                                          "invalid_target_rows": sum(i["split"] == split and i["is_target"] for i in issues)}
                                for split in EXPECTED},
                   "issues": issues}
        write_json(temporary / "data_quality.json", quality)
        write_json(temporary / "manifest.json", {"created_utc": utc_now(),
                   "purpose": "fresh_original_visdrone_conversion", "records": records,
                   "is_small_fixture": allow_small,
                   "geometry_policy": quality["policy"],
                   "data_quality_report": "data_quality.json",
                   "training_ignore_policy": "omit score=0/category=0/others from training labels; preserve raw for evaluation"})
        _write_dataset_yaml(temporary / "dataset.yaml", output)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Yeni veri hazir: {output}")
    print(f"Gecersiz kutu: {len(issues)}; hedef sinif satiri: {quality['invalid_target_rows']}")
    print(f"Veri kalite raporu: {output / 'data_quality.json'}")
    return output


def choose_capacity_records(records, n_train=24, n_val=8):
    """Only official TRAIN images: mix crowded images with the rest, no val tuning."""
    if n_train < 4 or n_val < 2:
        raise ValueError("Kapasite alt kumesi en az 4 train ve 2 val ornegi icermeli")
    candidates = sorted((r for r in records if r["split"] == "train" and r["gt_count"] > 0),
                        key=lambda r: (-r["gt_count"], r["image"]))
    if len(candidates) < n_train+n_val:
        raise ValueError(f"En az {n_train+n_val} pozitif orijinal train goruntusu gerekli")
    dense_count = (n_train+n_val)//2
    selected = candidates[:dense_count]
    remaining = candidates[dense_count:]
    needed = n_train+n_val-len(selected)
    selected += [remaining[(i*len(remaining))//needed] for i in range(needed)]
    # Alternate dense and representative images in each subset.
    mixed = []
    for index in range(max(dense_count, needed)):
        if index < dense_count:
            mixed.append(selected[index])
        if index < needed:
            mixed.append(selected[dense_count+index])
    return mixed[:n_train], mixed[n_train:n_train+n_val]


def make_capacity_subset(prepared, output, n_train=24, n_val=8):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot veri hedefi zaten var: {output}")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = choose_capacity_records(manifest["records"], n_train, n_val)
    output.mkdir(parents=True)
    chosen = []
    for split, selected in (("train", train), ("val", validation)):
        for record in selected:
            source = prepared / record["image"]
            if sha256(source) != record["image_sha256"]:
                raise ValueError(f"Hazirlanan goruntu degismis: {source.name}")
            _link_or_copy(source, output / "images" / split / source.name)
            labels = prepared / "labels" / "train" / (source.stem + ".txt")
            target = output / "labels" / split / labels.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(labels, target)
            chosen.append({**record, "capacity_split": split})
    write_json(output / "manifest.json", {"purpose": "capacity_only_not_scientific_results",
               "source_manifest_sha256": sha256(manifest_path), "records": chosen})
    _write_dataset_yaml(output / "dataset.yaml", output)
    return output / "dataset.yaml"
