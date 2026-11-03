"""Longer full-image stress pilot. Actual Stage-1 crops come in a later phase."""
from __future__ import annotations

import contextlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from .common import ROOT, sha256, utc_now, write_json
from .data import _link_or_copy, _write_dataset_yaml, choose_capacity_records, parse_annotation
from .pilot import ROLES


def stability_options(job):
    """Provisional stress recipe, explicitly independent of final research settings."""
    if job["epochs"] < 2 or job["batch"] < 1:
        raise ValueError("En az 2 epoch ve pozitif batch gerekli")
    if job["effective_batch"] < job["batch"] or job["effective_batch"] % job["batch"]:
        raise ValueError("Etkin batch fiziksel batch'in pozitif kati olmali")
    return dict(epochs=job["epochs"], nbs=job["effective_batch"], warmup_epochs=0.0,
                lr0=0.0001, lrf=1.0, weight_decay=0.0001, amp=True,
                mosaic=0.25, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                degrees=0.0, translate=0.1, scale=0.25, shear=0.0, perspective=0.0,
                flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
                multi_scale=False, workers=0, cache=False, rect=False)


def inspect_labels(text, source):
    """Exact five-value duplicate removal only; no IoU-based box merging."""
    rows, duplicates, seen = [], [], {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = tuple(float(v) for v in line.split())
        except ValueError as exc:
            raise ValueError(f"{source}, satir {line_number}: sayisal olmayan etiket") from exc
        if (len(row) != 5 or not all(math.isfinite(v) for v in row) or
                row[0] != int(row[0]) or not 0 <= row[0] <= 9):
            raise ValueError(f"{source}, satir {line_number}: sinif/etiket gecersiz")
        _, x, y, w, h = row
        if (w <= 0 or h <= 0 or x-w/2 < -1e-8 or y-h/2 < -1e-8 or
                x+w/2 > 1+1e-8 or y+h/2 > 1+1e-8):
            raise ValueError(f"{source}, satir {line_number}: normalize kutu gecersiz")
        if row in seen:
            duplicates.append({"line_number": line_number, "first_line_number": seen[row],
                               "values": list(row)})
        else:
            seen[row] = line_number
            rows.append(row)
    clean = "".join(f"{int(row[0])} " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                    for row in rows)
    return clean, duplicates, len(rows)


def build_stability_data(prepared, output, n_train=256, n_val=32):
    """Audit all prepared labels; sample disjoint TRAIN source images for both roles."""
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot veri hedefi zaten var: {output}")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "fresh_original_visdrone_conversion":
        raise ValueError("Temiz veri hazirlama manifesti gerekli")
    audited, cleaned, duplicate_issues = [], {}, []
    identities = set()
    for index, record in enumerate(manifest["records"]):
        split, relative = record["split"], PurePosixPath(record["image"])
        if (split not in {"train", "val"} or len(relative.parts) != 3 or
                relative.parts[:2] != ("images", split) or ".." in relative.parts or
                "\\" in str(relative)):
            raise ValueError("Manifest goruntu yolu gecersiz")
        identity = (split, relative.name.casefold())
        if identity in identities:
            raise ValueError("Manifestte ayni goruntu birden fazla kez var")
        identities.add(identity)
        label = prepared / "labels" / split / (relative.stem + ".txt")
        raw = prepared / "visdrone_annotations" / split / (relative.stem + ".txt")
        if sha256(raw) != record["annotation_sha256"]:
            raise ValueError(f"Ham annotation degismis: {raw}")
        label_text = label.read_text(encoding="utf-8")
        converted = parse_annotation(raw.read_text(encoding="utf-8-sig"),
                                     record["width"], record["height"], source=str(raw))
        expected = "".join(str(row[0]) + " " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                           for row in converted["labels"])
        if label_text != expected:
            raise ValueError(f"Hazir etiket ham annotation donusumuyle uyusmuyor: {label}")
        clean, duplicates, gt_count = inspect_labels(label_text, str(label))
        if gt_count + len(duplicates) != record["gt_count"]:
            raise ValueError(f"Etiket sayisi manifestle uyusmuyor: {label}")
        item = {**record, "gt_count": gt_count, "prepared_label_sha256": sha256(label),
                "duplicate_rows_removed_in_pilot": len(duplicates)}
        audited.append(item)
        cleaned[record["image"]] = clean
        if duplicates:
            duplicate_issues.append({"image": record["image"], "split": split,
                                     "source_label_sha256": item["prepared_label_sha256"],
                                     "duplicates": duplicates})
        if index % 500 == 0:
            print(f"Etiket denetimi: {index+1}/{len(manifest['records'])}", flush=True)
    train, validation = choose_capacity_records(audited, n_train, n_val)
    selected = []
    seen_hashes = set()
    output.mkdir(parents=True)
    try:
        for split, records in (("train", train), ("val", validation)):
            for record in records:
                source = prepared / record["image"]
                if sha256(source) != record["image_sha256"]:
                    raise ValueError(f"Goruntu degismis: {source}")
                if record["image_sha256"] in seen_hashes:
                    raise ValueError("Pilot alt kumelerinde ayni goruntu icerigi tekrar ediyor")
                seen_hashes.add(record["image_sha256"])
                destination = output / "images" / split / source.name
                _link_or_copy(source, destination)
                label = output / "labels" / split / (source.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(cleaned[record["image"]], encoding="utf-8")
                selected.append({**record, "pilot_split": split,
                                 "pilot_label_sha256": sha256(label)})
        report = {"created_utc": utc_now(), "audited_images": len(audited),
                  "duplicate_rows": sum(len(x["duplicates"]) for x in duplicate_issues),
                  "affected_images": len(duplicate_issues), "issues": duplicate_issues,
                  "policy": "exact class+normalized_xywh duplicates removed in pilot copies only; prepared/raw unchanged"}
        write_json(output / "label_audit.json", report)
        write_json(output / "manifest.json", {"purpose": "stability_full_image_only_not_scientific",
                   "source_manifest_sha256": sha256(manifest_path), "records": selected,
                   "source_split_policy": "official_train_only_for_both_pilot_splits",
                   "selection": "half dense, half uniformly spaced through remaining positives; disjoint source images",
                   "full_train_images": sum(r["split"] == "train" for r in audited),
                   "official_val_images": sum(r["split"] == "val" for r in audited)})
        if (prepared / "data_quality.json").exists():
            shutil.copy2(prepared / "data_quality.json", output / "source_data_quality.json")
        _write_dataset_yaml(output / "dataset.yaml", output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"Pilot verisi: {n_train} train + {n_val} kontrol; hepsi resmi TRAIN'den", flush=True)
    print(f"Tekrarlanan etiket: {report['duplicate_rows']} satir / {report['affected_images']} goruntu", flush=True)
    return output / "dataset.yaml"


def collect_child_result(job_dir, exit_code=None, timed_out=False):
    path = Path(job_dir) / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (ValueError, OSError):
        result = {}
    if timed_out:
        result["status"] = "TIMEOUT"
    elif not result or result.get("status") == "STARTED" or (
            exit_code != 0 and result.get("status") == "PASS_STABILITY_ONLY"):
        result["status"] = "PROCESS_FAILED"
    result["exit_code"] = exit_code
    return result


def bundle_reports(run_dir):
    """Allowlisted reports only; never data, weights or a Python environment."""
    run_dir = Path(run_dir)
    bundle = run_dir.with_name(run_dir.name + "_rapor.zip")
    allowed = {"summary.json", "OZET.txt", "job.json", "result.json", "console.log",
               "manifest.json", "label_audit.json", "source_data_quality.json", "epoch_progress.json"}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name in allowed:
                archive.write(path, path.relative_to(run_dir))
    return bundle


def run(prepared, reports, epochs=3, n_train=256, n_val=32, timeout=3600):
    from .__main__ import _Tee
    if epochs < 2 or timeout <= 0 or n_train % 8:
        raise ValueError("En az 2 epoch, pozitif timeout ve 8'in kati train sayisi gerekli")
    report_root = Path(reports).resolve()
    run_dir = report_root / ("stability_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True)
    results, manager_error = [], None
    dataset = None
    print(f"Kararlilik testi klasoru: {run_dir}", flush=True)
    with (run_dir / "console.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                dataset = build_stability_data(prepared, run_dir / "pilot_data", n_train, n_val)
                for role, batch in (("yolo800", 2), ("rtdetr800", 1), ("rtdetr960", 1)):
                    job_dir = run_dir / f"{role}_batch{batch}_amp"
                    job_dir.mkdir()
                    job = {**ROLES[role], "kind": "stability", "role": role, "batch": batch,
                           "precision": "amp", "epochs": epochs, "effective_batch": 8, "data": str(dataset)}
                    job_path = job_dir / "job.json"
                    write_json(job_path, job)
                    print(f"TEST: {role}, batch={batch}, epoch={epochs}. Gunluk: {job_dir / 'console.log'}", flush=True)
                    print("Egitim ayrintilari gunlukte; bu pencere tamamlaninca SONUC yazar.", flush=True)
                    started = time.perf_counter()
                    with (job_dir / "console.log").open("w", encoding="utf-8") as stream:
                        try:
                            completed = subprocess.run([sys.executable, "-m", "hycado_start", "_probe_child",
                                                        "--job", str(job_path)], cwd=ROOT, stdout=stream,
                                                       stderr=subprocess.STDOUT, timeout=timeout)
                            result = collect_child_result(job_dir, completed.returncode)
                        except subprocess.TimeoutExpired:
                            result = collect_child_result(job_dir, timed_out=True)
                            result["timeout_seconds"] = timeout
                    result.update(role=role, requested_batch=batch,
                                  parent_wall_seconds=time.perf_counter()-started)
                    write_json(job_dir / "result.json", result)
                    results.append(result)
                    print(f"SONUC: {role}: {result['status']}", flush=True)
                    write_json(run_dir / "summary.json", {"purpose": "stability_only", "results": results})
            except Exception as exc:
                manager_error = str(exc)
                traceback.print_exc()
            finally:
                write_json(run_dir / "summary.json", {"purpose": "stability_only", "results": results,
                           "manager_error": manager_error, "epochs": epochs,
                           "n_train": n_train, "n_val_from_train": n_val,
                           "actual_stage1_crops_tested": False,
                           "recipe_is_provisional": True})
                lines = ["HyCADO uzun yerel kararlilik pilotu", "",
                         "Bu pilot bilimsel sonuc veya nihai egitim degildir.",
                         "RT-DETR 960 tam goruntu vekilidir; gercek Stage-1 crop'lari henuz test edilmedi.",
                         "Artirmalar ve etkin batch=8 deneme protokoludur; nihai tarif degildir.",
                         "PyTorch bellegi diger uygulamalarin GPU bellegini kapsamaz.", ""]
                for result in results:
                    lines.append(f"{result['role']} batch={result['requested_batch']}: {result['status']}; "
                                 f"peak_reserved_GiB={result.get('peak_reserved_gib', 'NA')}; "
                                 f"actual_optimizer_steps={result.get('actual_optimizer_steps', 'NA')}")
                if manager_error:
                    lines.append("Yonetici hatasi: " + manager_error)
                lines.extend(["", "Sureyi epoch_records icindeki veri yukleme dahil train_seconds ve validation dahil epoch_seconds ile degerlendirin.",
                              "Kucuk, yogunluk secimli alt kumeden tam egitim suresi kesin olarak hesaplanamaz."])
                (run_dir / "OZET.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle = bundle_reports(run_dir)
    print(f"GONDERILECEK RAPOR: {bundle}", flush=True)
    return bundle, manager_error is None
