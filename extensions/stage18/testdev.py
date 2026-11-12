"""VisDrone DET test-dev validation of the frozen HyCADO v1 protocol."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import zlib
from contextlib import closing

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))
from hycado_start.common import sha256, utc_now, write_json
from hycado_start.stage1 import fingerprint, training_lock

FROZEN = HERE / "evidence/frozen.json"
COUNT = 1610
PILOT_N = 12
DRAWS = 1000
DRAW_SEED = 20261102
DATASET = "VisDrone2019-DET-test-dev"
RUN = "testdev_frozen_v1_seed42"
METHODS = ("yolo800", "yolo1280", "hybrid_a025", *(f"relocated_s{s}_a025" for s in range(3)))
CROP_METHODS = METHODS[2:]
REPORT = "18_testdev_dogrulama_rapor.zip"
PREFLIGHT = "18_testdev_onkontrol.json"
STEM = re.compile(r"[A-Za-z0-9_-]+")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")
    require(sha256(path) == expected, f"Dosya SHA256 eslesmedi: {path}")


def safe(root, rel):
    parts = PurePosixPath(rel).parts
    require(parts and all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in {".", ".."} for p in parts),
            f"Gecersiz dosya yolu: {rel}")
    path = root.joinpath(*parts).resolve()
    require(path.is_relative_to(root.resolve()), f"Dosya klasor disina cikiyor: {rel}")
    return path


def locate_dataset(root, explicit):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        require(candidate.is_dir() or candidate.is_file(), f"Test-dev girdisi bulunamadi: {candidate}")
        return candidate
    locations = (HERE, HERE.parent, Path.home() / "Downloads", root / "data" / "raw", root / "data", root)
    for folder in dict.fromkeys(locations):
        for name in (DATASET + ".zip", DATASET):
            path = folder / name
            if path.is_file() or path.is_dir():
                return path.resolve()
        if folder.is_dir():
            variants = sorted(folder.glob(DATASET + " (*).zip"))
            if len(variants) == 1:
                return variants[0].resolve()
            if len(variants) > 1:
                raise ValueError(f"Birden fazla test-dev ZIP var: {folder}. Ikinci argumanla dosyayi belirtin.")
    raise FileNotFoundError(f"{DATASET}.zip bulunamadi. Resmi ZIP'i CMD dosyasinin yanina koyun.")


def pair_check(images, labels, count):
    require(len(images) == len(labels) == count and set(images) == set(labels),
            f"Test-dev: {count} benzersiz JPG/TXT cifti olmali; JPG={len(images)}, TXT={len(labels)}.")
    require(len({name.casefold() for name in images}) == len(images), "Ayni isimli JPG/TXT kayitlari var.")


def dataset_dir(src, out):
    if src.is_dir():
        if (src / DATASET / "images").is_dir():
            src = src / DATASET
        return src, None
    require(src.suffix.lower() == ".zip", "Test-dev girdisi bir ZIP veya acilmis klasor olmali.")
    archive_sha = sha256(src)
    target = out / ("dataset_" + archive_sha[:16])
    if target.is_dir():
        require(read(target / "archive.json") == {"sha256": archive_sha}, "Acilmis test-dev ZIP kimligi farkli.")
        return target, archive_sha
    with zipfile.ZipFile(src) as archive:
        found = {"images": {}, "annotations": {}}
        prefix = set()
        for member in archive.infolist():
            if member.is_dir():
                continue
            require("\\" not in member.filename, "ZIP icinde gecersiz yol.")
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 2 or parts[-2] not in found:
                continue
            kind, name = parts[-2:]
            ext = ".jpg" if kind == "images" else ".txt"
            stem = Path(name).stem
            require(len(parts) in (2, 3) and (len(parts) == 2 or parts[0] == DATASET) and
                    Path(name).suffix.lower() == ext and STEM.fullmatch(stem) and
                    stem not in found[kind] and 0 <= member.file_size <= (32 << 20),
                    f"Beklenmedik test-dev ZIP kaydi: {member.filename}")
            prefix.add(parts[:-2])
            found[kind][stem] = (name, member)
        require(len(prefix) == 1, "ZIP'te birden fazla veri koku var.")
        pair_check(found["images"], found["annotations"], COUNT)
        require(sum(m.file_size for items in found.values() for _, m in items.values()) <= (2 << 30),
                "Test-dev ZIP acilmis boyutu beklenenden buyuk.")
        with tempfile.TemporaryDirectory(dir=out, prefix="extract_") as work:
            staging = Path(work) / "dataset"
            for kind, items in found.items():
                (staging / kind).mkdir(parents=True)
                for name, member in items.values():
                    with archive.open(member) as raw, (staging / kind / name).open("xb") as dest:
                        shutil.copyfileobj(raw, dest)
            write_json(staging / "archive.json", {"sha256": archive_sha})
            staging.rename(target)
    return target, archive_sha


def scan_dataset(path, prepared, archive_sha):
    from PIL import Image
    def by_stem(folder, ext):
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext]
        result = {p.stem: p for p in files}
        require(len(result) == len(files), f"Ayni isimli dosyalar var: {folder}")
        return result
    images = by_stem(path / "images", ".jpg")
    labels = by_stem(path / "annotations", ".txt")
    pair_check(images, labels, COUNT)
    require(all(STEM.fullmatch(x) for x in images), "Test-dev dosya adi beklenmeyen karakter iceriyor.")
    old_names = {Path(r["image"]).stem.casefold() for r in prepared["records"]}
    old_hashes = {r["image_sha256"] for r in prepared["records"]}
    records = []
    for i, name in enumerate(sorted(images), 1):
        image = images[name]
        require(image.resolve().is_relative_to(path.resolve()) and labels[name].resolve().is_relative_to(path.resolve()),
                "Veri klasoru disina isaret eden JPG/TXT dosyasi var.")
        stat = labels[name].stat()  # Do not open annotation rows before the inference freeze.
        with Image.open(image) as im:
            w, h = im.size
            require(im.format == "JPEG" and w > 0 and h > 0, f"JPEG boyutu/bicimi gecersiz: {image}")
        digest = sha256(image)
        require(name.casefold() not in old_names and digest not in old_hashes,
                f"TRAIN/VAL ve test-dev arasinda ayni dosya adi veya JPEG SHA256: {name}")
        records.append({"image": f"images/test-dev/{image.name}", "stem": name, "width": w, "height": h,
                        "image_sha256": digest, "annotation": labels[name].name,
                        "annotation_size": stat.st_size, "annotation_mtime_ns": stat.st_mtime_ns})
        if i == 1 or i % 400 == 0 or i == COUNT:
            print(f"TEST-DEV veri kimligi: {i}/{COUNT}", flush=True)
    return {"split": DATASET, "official_zip_sha256": archive_sha, "records": records}


def scan_project(root):
    f = read(FROZEN)
    for name, digest in f["engine_sha256"].items():
        check_file(safe(HERE / "engine/hycado_start", name), digest)
    require(f["crop_score_multiplier"] == .25 and f["fusion"]["nms_iou"] == .7 and
            f["fusion"]["max_det"] == 1000 and f["crop_inference"]["imgsz"] == 960,
            "Beklenen sabit tarif degismis.")
    for name, version in f["runtime_versions"].items():
        require(importlib.metadata.version(name) == version,
                f"Paket surumu farkli: {name}; gereken {version}. Projenin .venv ortamiyla calistirin.")
    experiments = root / "experiments"
    old = experiments / "official_val_a025_v1_seed42"
    p2 = experiments / "p2_strong_baselines_v1_seed42"
    check_file(old / "protocol.json", f["original_protocol_file_sha256"])
    check_file(p2 / "official_val_protocol.json", f["p2_protocol_file_sha256"])
    prepared = root / "data/prepared/manifest.json"
    check_file(prepared, f["prepared_manifest_sha256"])
    train_val = read(prepared)
    require(sum(r["split"] == "train" for r in train_val["records"]) == 6471 and
            sum(r["split"] == "val" for r in train_val["records"]) == 548,
            "Dondurulmus orijinal TRAIN/VAL kapsaminda fark var.")
    weights = {}
    for kind, name, key in [("yolo", "stage1_v1_seed42", "stage1_protocol_sha256"),
                            ("crop", "rtdetr_crop_v1_seed42", "crop_training_protocol_sha256")]:
        folder = experiments / name
        proto = read(folder / "protocol.json")
        require(fingerprint(proto) == f[key], f"Orijinal egitim protokolu farkli: {name}")
        weight = root / "release_stage" / "checkpoints" / f"{kind}_epoch_0080.pt"
        if not weight.is_file():
            weight = safe(folder, read(folder / "checkpoint_index.json")["latest"]["path"])
        check_file(weight, f["checkpoint_sha256"][kind])
        weights[kind] = weight
        if kind == "crop":
            official = root / "work/fresh_official_pretrained/rtdetr-l.pt"
            check_file(official, proto["initial_weights_sha256"])
            weights["official"] = official
            f["architecture_checkpoint_sha256"] = proto["initial_weights_sha256"]
    return f, train_val, weights, old, p2


def environment():
    import torch
    require(torch.cuda.is_available(), "CUDA GPU bulunamadi; CPU'ya gecilmedi.")
    prop = torch.cuda.get_device_properties(0)
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True, timeout=15)
    return {"gpu": prop.name, "vram_bytes": prop.total_memory, "cuda": torch.version.cuda,
            "driver_inventory": probe.stdout.strip(), "python": list(sys.version_info[:3])}


def check_labels_stat(rec, labels):
    stat = labels.stat()
    require((stat.st_size, stat.st_mtime_ns) == (rec["annotation_size"], rec["annotation_mtime_ns"]),
            f"Annotation dosya bilgisi degismis: {labels}")


def val_references(project, f, old, p2):
    from hycado_start.frozen_val import checked_json
    ans = []
    for item in f["val_parity"]:
        rec = item["record"]
        image = safe(project / "data/prepared", rec["image"])
        check_file(image, rec["image_sha256"])
        frozen = safe(old, item["original"]["path"])
        check_file(frozen, item["original"]["sha256"])
        reference = checked_json(frozen, f["original_protocol_content_sha256"], rec)
        entry = {}
        for method in METHODS[:2]:
            path = safe(p2, item[method]["path"])
            check_file(path, item[method]["sha256"])
            payload = read(path)
            require(payload["protocol_sha256"] == f["p2_protocol_content_sha256"] and
                    payload["image_sha256"] == rec["image_sha256"], f"P2 VAL kimligi farkli: {path}")
            entry[method] = payload["detections"]
        require(entry["yolo800"] == reference["baseline"], "12 ve P2 YOLO-800 referanslari farkli.")
        ans.append((rec, image, entry, reference))
    return ans


def open_cache(path, signature):
    db = sqlite3.connect(path, timeout=10)
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS prediction (method TEXT, stem TEXT, digest TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(method,stem))")
            db.execute("CREATE TABLE IF NOT EXISTS matching (stem TEXT PRIMARY KEY, ann_sha TEXT NOT NULL, digest TEXT NOT NULL, payload BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS bootstrap (draw INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO binding VALUES(1,?)", (signature,))
            require(db.execute("SELECT signature FROM binding WHERE id=1").fetchone() == (signature,),
                    "SQL tahmin kasasinin protokolu farkli.")
        return db
    except BaseException:
        db.close()
        raise


def cache_rows(db, method, rec):
    item = db.execute("SELECT digest,payload FROM prediction WHERE method=? AND stem=?", (method, rec["stem"])).fetchone()
    if item is None:
        return None
    digest, encoded = item
    raw = zlib.decompress(encoded)
    require(hashlib.sha256(raw).hexdigest() == digest, f"Tahmin kaydi bozulmus: {method}/{rec['stem']}")
    body = json.loads(raw)
    require(body["image_sha256"] == rec["image_sha256"], f"Tahmin goruntu baglantisi farkli: {rec['stem']}")
    return body


def cache_prediction(db, method, rec, rows, dropped=0, rects=None, baseline_sha=None):
    from hycado_start.proposal_geometry import canonical_detections
    clean, invalid = canonical_detections(rows, rec["width"], rec["height"])
    require(not invalid and clean == rows and len(rows) <= 1000, f"Gecersiz tahmin: {method}/{rec['stem']}")
    body = {"image_sha256": rec["image_sha256"], "detections": rows, "dropped_zero_area": dropped,
            "rects": rects, "baseline_sha256": baseline_sha}
    raw = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    with db:
        db.execute("INSERT INTO prediction VALUES(?,?,?,?)",
                   (method, rec["stem"], hashlib.sha256(raw).hexdigest(), zlib.compress(raw, 1)))


def image_path(dataset, rec):
    path = dataset / "images" / PurePosixPath(rec["image"]).name
    check_file(path, rec["image_sha256"])
    return path


def yolo_phase(method, f, weights, out, dataset, records, db, parity):
    import torch
    from hycado_start import proposals
    from hycado_start.runtime_benchmark import backend_state, restore_backend
    state = backend_state()
    pending = [rec for rec in records if cache_rows(db, method, rec) is None]
    if not pending and not parity:
        return []
    config = {"inference": {**f["stage1_inference"], "imgsz": int(method[4:])}}
    predict = proposals.make_predictor(proposals.load_ema(weights["yolo"], f["stage1_protocol_sha256"]), config, out)
    try:
        for rec, path, expected, _ in parity:
            w, h, rows, _ = predict(path)
            require((w, h) == (rec["width"], rec["height"]) and rows == expected[method],
                    f"VAL birebir eslik basarisiz: {method}/{rec['image']}")
        print(f"TEST-DEV {method}: {len(records)-len(pending)} kayit korunuyor, {len(pending)} yeni.", flush=True)
        timings = []
        for i, rec in enumerate(pending, 1):
            start = time.perf_counter()
            w, h, rows, dropped = predict(image_path(dataset, rec))
            require((w, h) == (rec["width"], rec["height"]), f"Goruntu boyutu farkli: {rec['image']}")
            cache_prediction(db, method, rec, rows, int(dropped))
            timings.append(time.perf_counter() - start)
            progress(out, method, len(records)-len(pending)+i, len(records), i)
        return timings
    finally:
        del predict
        gc.collect()
        torch.cuda.empty_cache()
        restore_backend(state)


def ordered_rects(rec, baseline, f):
    from hycado_start.crop_data import select_clusters
    from hycado_start.position_control import relocate, SEEDS
    _, clusters = select_clusters({"width": rec["width"], "height": rec["height"], "detections": baseline}, f["crop_policy"])
    rects = [list(c["rect"]) for c in clusters]
    plan = {"hybrid_a025": rects}
    for seed in SEEDS:
        result = relocate(rects, rec["width"], rec["height"], rec["image"], seed)
        require([(x2-x1, y2-y1) for x1, y1, x2, y2 in rects] ==
                [(x2-x1, y2-y1) for x1, y1, x2, y2 in result], "Crop sayisi veya boyutu farkli.")
        plan[f"relocated_s{seed}_a025"] = result
    return plan


def crop_rows(model, rgb, rec, baseline, rects):
    import numpy as np
    from hycado_start.dev_evaluation import map_queries
    from hycado_start.nms_equivalence import fast_nms
    from hycado_start.runtime_benchmark import direct_queries
    candidates = list(baseline)
    for rect in rects:
        rows = map_queries(direct_queries(model, np.array(rgb.crop(tuple(rect)))), {"source_crop_xyxy": rect})
        for row in rows:
            row[4] *= .25
        candidates.extend(rows)
    return fast_nms(candidates, rec["width"], rec["height"])


def rgb_image(path, rec):
    from PIL import Image
    with Image.open(path) as image:
        require(image.size == (rec["width"], rec["height"]), f"RGB goruntu boyutu farkli: {path}")
        return image.convert("RGB")


def crop_phase(f, weights, out, dataset, records, db, parity):
    import torch
    from hycado_start.crop_capture import load_model
    from hycado_start.runtime_benchmark import backend_state, restore_backend
    pending = [(rec, [name for name in CROP_METHODS if cache_rows(db, name, rec) is None]) for rec in records]
    pending = [(rec, need) for rec, need in pending if need]
    if not pending and not parity:
        return [], []
    state = backend_state()
    model = load_model(weights["crop"], weights["official"], "cuda")
    try:
        for rec, path, expected, reference in parity:
            baseline = expected["yolo800"]
            rects = ordered_rects(rec, baseline, f)["hybrid_a025"]
            require(rects == reference["crop_rects"], f"VAL crop geometrisi eslesmedi: {rec['image']}")
            rows = crop_rows(model, rgb_image(path, rec), rec, baseline, rects)
            require(rows == reference["hybrid_a025"], f"VAL HyCADO birebir eslik basarisiz: {rec['image']}")
        # On new images, inference does not receive or open any annotation.
        print(f"TEST-DEV crop: {len(records)-len(pending)} tam goruntu korunuyor, {len(pending)} eksik.", flush=True)
        timings, counts = [], []
        for i, (rec, need) in enumerate(pending, 1):
            start = time.perf_counter()
            base = cache_rows(db, "yolo800", rec)
            require(base is not None, "Crop icin sabit YOLO-800 tahmini eksik.")
            baseline = base["detections"]
            baseline_sha = fingerprint(baseline)
            plan = ordered_rects(rec, baseline, f)
            rgb = rgb_image(image_path(dataset, rec), rec)
            # Identical rectangles (including coincidences under relocation) have identical model inputs.
            # Keep the original batch-one forwards for every crop and variant, without cross-variant reuse.
            for name in need:
                rows = crop_rows(model, rgb, rec, baseline, plan[name])
                cache_prediction(db, name, rec, rows, rects=plan[name], baseline_sha=baseline_sha)
            timings.append(time.perf_counter()-start)
            counts.append(sum(len(plan[name]) for name in need))
            progress(out, "crop_4_variants", len(records)-len(pending)+i, len(records), i)
        return timings, counts
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        restore_backend(state)


def progress(out, phase, done, total, new):
    if new == 1 or done % 25 == 0 or done == total:
        print(f"TEST-DEV {phase}: {done}/{total}", flush=True)
        write_json(out / "progress.json", {"phase": phase, "done": done, "total": total, "utc": utc_now()})


def prediction_freeze(db, records, f, signature, out):
    entries = []
    require(db.execute("SELECT COUNT(*) FROM prediction").fetchone()[0] == len(records)*len(METHODS),
            "Tum tahminler tamamlanmadan test-dev etiketleri acilmaz.")
    for rec in records:
        rows = {method: cache_rows(db, method, rec) for method in METHODS}
        require(all(x is not None for x in rows.values()), f"Yontem/tahmin eksik: {rec['stem']}")
        base = rows["yolo800"]["detections"]
        base_sha = fingerprint(base)
        plans = ordered_rects(rec, base, f)
        for method in CROP_METHODS:
            require(rows[method]["rects"] == plans[method] and rows[method]["baseline_sha256"] == base_sha,
                    f"Crop geometrisi veya baseline baglantisi farkli: {rec['stem']}/{method}")
        entries.append({"image": rec["image"], "crop_count_each_variant": len(plans["hybrid_a025"]),
                        "variant_sha256": dict(db.execute("SELECT method,digest FROM prediction WHERE stem=? ORDER BY method", (rec["stem"],)))})
    index = {"protocol_sha256": signature, "records": entries}
    path = out / "prediction_index.json"
    if path.exists():
        require(read(path) == index, "Dondurulmus tahmin indeksi degismis.")
    else:
        write_json(path, index)
    return index


def metric_part(rec, payload, rows):
    import numpy as np
    from hycado_start import dev_metrics as m, score_ablation as s
    raw = m.raw_annotation(payload.decode("utf-8-sig"))
    integral = m.ignore_integral(raw, rec["width"], rec["height"])
    gt = m.prepare_groundtruth(raw, integral)
    part = {"gt_denominators": np.array([np.count_nonzero(gt[:, 5] == c) for c in range(1, 11)], dtype=np.int64)}
    part["presence"] = (part["gt_denominators"] > 0).astype(np.int64)
    for method in METHODS:
        det = m.outside_ignore(m.xyxy_to_det(rows[method]), integral)[:500]
        part[method+"_scores"] = det[:, 4].copy()
        part[method+"_classes"] = det[:, 5].astype(np.int8)
        part[method+"_flags"] = np.column_stack([s.fast_flags(det, gt, float(t)) for t in s.THRESHOLDS])
    return part


def score(db, records, dataset, signature, index, out):
    import numpy as np
    from hycado_start.score_ablation import aggregate
    from hycado_start.dev_metrics import REFERENCE_COMMIT
    parts, label_index = [], []
    for i, rec in enumerate(records, 1):
        path = dataset / "annotations" / rec["annotation"]
        check_labels_stat(rec, path)
        payload = path.read_bytes()
        ann_sha = hashlib.sha256(payload).hexdigest()
        check_labels_stat(rec, path)
        old = db.execute("SELECT ann_sha,digest,payload FROM matching WHERE stem=?", (rec["stem"],)).fetchone()
        if old:
            require(old[0] == ann_sha and hashlib.sha256(old[2]).hexdigest() == old[1],
                    f"Etiket veya kayitli eslesmeler degismis: {rec['stem']}")
            with np.load(io.BytesIO(old[2]), allow_pickle=False) as data:
                part = {k: data[k] for k in data.files}
        else:
            rows = {name: cache_rows(db, name, rec)["detections"] for name in METHODS}
            part = metric_part(rec, payload, rows)
            stream = io.BytesIO()
            np.savez_compressed(stream, **part)
            encoded = stream.getvalue()
            with db:
                db.execute("INSERT INTO matching VALUES(?,?,?,?)",
                           (rec["stem"], ann_sha, hashlib.sha256(encoded).hexdigest(), encoded))
        parts.append(part)
        label_index.append({"stem": rec["stem"], "sha256": ann_sha})
        progress(out, "METRIK", i, len(records), i)
    write_json(out / "annotation_index.json", {"protocol_sha256": signature, "records": label_index})
    metrics = {method: aggregate(parts, method) for method in METHODS}
    keys = ["AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500"]
    mean = {k: sum(metrics[m][k] for m in CROP_METHODS[1:])/3 for k in keys}
    diff = {k: metrics["hybrid_a025"][k]-mean[k] for k in keys}
    point = {"status": "POINT_METRICS_COMPLETE", "images": len(records), "protocol_sha256": signature,
             "metric": "pinned VisDrone DET toolkit-aligned Python implementation, percentage points",
             "reference_commit": REFERENCE_COMMIT, "separate_testdev_reference_binary_run": False,
             "methods": metrics, "relocation_mean": mean, "hycado_minus_relocation_mean_pp": diff,
             "hycado_minus_yolo800_AP_pp": metrics["hybrid_a025"]["AP"]-metrics["yolo800"]["AP"],
             "hycado_minus_yolo1280_AP_pp": metrics["hybrid_a025"]["AP"]-metrics["yolo1280"]["AP"],
             "crop_calls_each_crop_method": sum(row["crop_count_each_variant"] for row in index["records"])}
    write_json(out / "metrics.json", point)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["method", *keys])
        for name in METHODS:
            writer.writerow([name, *(metrics[name][k] for k in keys)])
        writer.writerow(["relocated_mean", *(mean[k] for k in keys)])
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *metrics[METHODS[0]]["per_class"][0]])
        writer.writeheader()
        for name in METHODS:
            writer.writerows({"method": name, **row} for row in metrics[name]["per_class"])
    print(f"TEST-DEV nokta sonucu hazir: HyCADO - relocation ort. = {diff['AP']:+.3f} AP", flush=True)
    return point, parts


def bootstrap(db, records, parts, point, out):
    import numpy as np
    from hycado_start import bootstrap_statistics as b
    groups = b.make_groups(records)
    write_json(out / "bootstrap_groups.json", groups)
    require(len(groups["labels"]) >= 2, "Bootstrap icin en az iki dosya adi grubu gerekli.")
    engine = b.compile_engine(parts, CROP_METHODS)
    orig = b.evaluate(engine, np.ones(len(records), dtype=np.int64))
    expected = np.array([[point["methods"][method][key] for key in b.KEYS] for method in CROP_METHODS])
    require(np.allclose(orig, expected, atol=1e-8, rtol=0), "Bootstrap ve ana metrik nokta esligi basarisiz.")
    completed = dict(db.execute("SELECT draw,payload FROM bootstrap ORDER BY draw"))
    require(all(type(i) is int and 0 <= i < DRAWS for i in completed), "Bootstrap kaydi kapsam disi.")
    print(f"TEST-DEV CPU bootstrap: {len(completed)}/{DRAWS} korunuyor.", flush=True)
    draws = []
    for i in range(DRAWS):
        if i in completed:
            envelope = json.loads(completed[i])
            require(envelope["sha256"] == fingerprint({"draw": i, "value": envelope["value"]}),
                    f"Bootstrap kaydi bozulmus: {i}")
            value = np.asarray(envelope["value"], dtype=float) if envelope["value"] is not None else np.full(orig.shape, np.nan)
            require(value.shape == orig.shape and (envelope["value"] is None or np.isfinite(value).all()),
                    f"Bootstrap kaydi bicimi farkli: {i}")
        else:
            value = b.evaluate(engine, b.frequencies(groups, "cluster", i, seed=DRAW_SEED))
            payload = value.tolist() if np.isfinite(value).all() else None
            envelope = {"value": payload, "sha256": fingerprint({"draw": i, "value": payload})}
            with db:
                db.execute("INSERT INTO bootstrap VALUES(?,?)", (i, json.dumps(envelope, allow_nan=False)))
            progress(out, "BOOTSTRAP", i+1, DRAWS, i+1)
        draws.append(value)
    results = b.intervals(orig, np.stack(draws), CROP_METHODS)
    results["primary_exploratory_AP_contrasts"] = ["hybrid_minus_relocated_mean"]
    results["protocol_sha256"] = point["protocol_sha256"]
    results["sampling_scope"] = "paired filename-prefix/hash proxy clusters, not verified independent scenes"
    results["group_count"] = len(groups["labels"])
    results["draw_seed"] = DRAW_SEED
    write_json(out / "intervals.json", results)
    return results["differences_pp"]["hybrid_minus_relocated_mean"]["AP"]


def bind_protocol(out, name, body):
    path = out / name
    if path.exists():
        require(read(path) == body, f"{name} farkli. Eski ve yeni kosular karistirilmadi.")
    else:
        write_json(path, body)


def make_report(out):
    with tempfile.TemporaryDirectory(dir=out) as work:
        dest = Path(work) / REPORT
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in ["result.json", "protocol.json", PREFLIGHT, "dataset_manifest.json", "prediction_index.json",
                         "annotation_index.json", "heldout_confirmation.json", "metrics.json", "metrics.csv", "per_class.csv",
                         "intervals.json", "bootstrap_groups.json"]:
                z.write(out / name, arcname=name)
        os.replace(dest, out / REPORT)


def run(args, root, out):
    if args.mode == "full":
        require(args.heldout == "EVET", "Test-dev'in ayar/model secimi icin kullanilmadigini dogrulayin.")
    src = locate_dataset(root, args.dataset)
    f, prepared, weights, old, p2 = scan_project(root)
    env = environment()
    dataset, archive_sha = dataset_dir(src, out)
    manifest = scan_dataset(dataset, prepared, archive_sha)
    records = manifest["records"]
    bind_protocol(out, "dataset_manifest.json", manifest)
    protocol = {"id": RUN, "dataset_manifest_sha256": fingerprint(manifest), "images": COUNT,
                "frozen_evidence_sha256": sha256(FROZEN), "runner_sha256": sha256(Path(__file__)),
                "upstream_checkpoint_sha256": f["checkpoint_sha256"] | {"architecture": f["architecture_checkpoint_sha256"]},
                "prepared_train_val_manifest_sha256": f["prepared_manifest_sha256"], "environment": env,
                "methods": list(METHODS), "relocation_seeds": [0, 1, 2],
                "yolo_input_sizes": [800, 1280], "yolo_800_settings": f["stage1_inference"],
                "crop_inference": f["crop_inference"], "crop_policy": f["crop_policy"],
                "fusion": f["fusion"], "crop_score_multiplier": .25,
                "primary_endpoint": "toolkit AP of hybrid minus arithmetic mean of the three separate relocation APs",
                "bootstrap": {"draws": DRAWS, "seed": DRAW_SEED, "mode": "paired proxy cluster percentile",
                              "scope": "fixed model weights and fixed location seeds"},
                "training": False, "parameter_search": False,
                "gt_policy": "No test-dev annotation rows opened until all methods are durably frozen.",
                "overlap_check": "same stem (case-insensitive) or JPEG bytes SHA256 against prepared TRAIN and VAL",
                "pilot": "deterministic subset, no test annotation access, inference outputs reused in full run"}
    bind_protocol(out, "protocol.json", protocol)
    signature = fingerprint(protocol)
    preview = out / PREFLIGHT
    if args.mode == "full":
        pilot = read(preview)
        require(pilot.get("status") == "PREFLIGHT_OK" and pilot.get("protocol_sha256") == signature,
                "Once 18A on kontrolunu basariyla tamamlayin.")
        attestation = out / "heldout_confirmation.json"
        if (out / REPORT).is_file() and (out / "result.json").is_file():
            final = read(out / "result.json")
            require(final.get("status") == "COMPLETED" and final.get("protocol_sha256") == signature,
                    "Tamamlanmis rapor farkli bir kosuya ait.")
            check_file(out / "predictions.sqlite3", final["prediction_database_sha256"])
            print(f"SONUC: COMPLETED (mevcut)\nGONDERILECEK RAPOR: {out / REPORT}", flush=True)
            return
        if attestation.exists():
            require(read(attestation)["protocol_sha256"] == signature and read(attestation)["heldout_user_confirmed"] is True,
                    "Test-dev kullanim beyani farkli.")
        else:
            write_json(attestation, {"heldout_user_confirmed": True, "protocol_sha256": signature, "utc": utc_now()})
    elif preview.exists():
        pilot = read(preview)
        if pilot.get("status") == "PREFLIGHT_OK" and pilot.get("protocol_sha256") == signature:
            print(f"ON KONTROL daha once tamamlandi: {preview}", flush=True)
            print(f"Tahmini GPU/cikarim suresi: {pilot['estimated_inference_hours']:.2f} saat", flush=True)
            return
    if args.mode == "preflight":
        parity = val_references(root, f, old, p2)
        active = sorted(records, key=lambda rec: fingerprint({"testdev_pilot": rec["image"]}))[:PILOT_N]
        # Time a fresh pilot even after an interrupted preview. Full-run cache is never cleared.
        cache_path = out / "pilot.sqlite3"
        for suffix in ("", "-journal"):
            (out / (cache_path.name + suffix)).unlink(missing_ok=True)
    else:
        parity = []
        active = records
        cache_path = out / "predictions.sqlite3"
        require((out / "pilot.sqlite3").is_file(), "Basarili 18A pilot SQLite dosyasi bulunamadi.")

    with closing(open_cache(cache_path, signature)) as db:
        if args.mode == "full":
            with closing(open_cache(out / "pilot.sqlite3", signature)) as src_db:
                sample = sorted(records, key=lambda rec: fingerprint({"testdev_pilot": rec["image"]}))[:PILOT_N]
                for rec in sample:
                    for method in METHODS:
                        b = cache_rows(src_db, method, rec)
                        require(b is not None, f"Pilot tahmini eksik: {method}/{rec['stem']}")
                        if cache_rows(db, method, rec) is None:
                            cache_prediction(db, method, rec, b["detections"], b["dropped_zero_area"], b["rects"], b["baseline_sha256"])
        totals = {method: yolo_phase(method, f, weights, out, dataset, active, db, parity) for method in METHODS[:2]}
        crop_times, counts = crop_phase(f, weights, out, dataset, active, db, parity)
        if args.mode == "preflight":
            require(all(len(x) == PILOT_N for x in [*totals.values(), crop_times, counts]), "Pilot sure kapsami eksik.")
            secs = sum(sum(x) for x in [*totals.values(), crop_times])
            hours = secs / PILOT_N * COUNT / 3600
            write_json(preview, {"status": "PREFLIGHT_OK", "protocol_sha256": signature, "val_exact_images": len(parity),
                                 "pilot_images": [r["image"] for r in active], "testdev_gt_parsed": False,
                                 "estimated_inference_hours": hours, "pilot_inference_seconds": secs,
                                 "pilot_crop_forwards_all_four_methods": sum(counts),
                                 "pilot_seconds_by_phase": {k: sum(v) for k, v in totals.items()} | {"crop_variants": sum(crop_times)},
                                 "estimate_scope": "approximate only, excludes CPU metric/bootstrap and setup; not a benchmark"})
            print(f"ON KONTROL TAMAM. Tahmini GPU/cikarim suresi: {hours:.2f} saat (CPU metrik ve GA haric).", flush=True)
            print(f"RAPOR: {preview}", flush=True)
            return
        index = prediction_freeze(db, records, f, signature, out)
        point, parts = score(db, records, dataset, signature, index, out)
        ci = bootstrap(db, records, parts, point, out)
    write_json(out / "result.json", {"status": "COMPLETED", "protocol_sha256": signature,
                "images": COUNT, "methods": list(METHODS), "heldout_user_confirmed": True,
                "primary_AP_difference_pp": ci, "crop_calls_each_crop_method": point["crop_calls_each_crop_method"],
                "test_dev_ground_truth_read_after_prediction_freeze": True, "training": False,
                "multiple_training_seeds": False, "latency_benchmark": False,
                "prediction_database_sha256": sha256(cache_path),
                "heldout_attestation_sha256": sha256(attestation), "completed_utc": utc_now()})
    make_report(out)
    print(f"SONUC: COMPLETED\nGONDERILECEK RAPOR: {out / REPORT}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "full"))
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--dataset", default=None)
    p.add_argument("--heldout", default="")
    args = p.parse_args()
    root = args.project.resolve()
    if not (root / "data/prepared").is_dir() or not (root / "experiments").is_dir():
        print(f"Gecersiz HyCADO proje koku: {root}", file=sys.stderr)
        return 1
    out = root / "experiments" / RUN
    out.mkdir(exist_ok=True)
    try:
        with training_lock(out / "run.lock"):
            try:
                run(args, root, out)
                (out / f"{args.mode}_error.json").unlink(missing_ok=True)
            except (Exception, KeyboardInterrupt) as exc:
                write_json(out / f"{args.mode}_error.json", {"status": "STOPPED", "phase": args.mode,
                           "error_type": type(exc).__name__, "error": str(exc),
                           "traceback": traceback.format_exc(), "utc": utc_now()})
                raise
        return 0
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        print(f"DURDU. Hata raporu (yazilabildiyse): {out / (args.mode + '_error.json')}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
