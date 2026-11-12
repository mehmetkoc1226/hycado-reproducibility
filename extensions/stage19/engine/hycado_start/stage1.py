"""First real research run: fixed-budget YOLOv9-C with auditable epoch resume."""
from __future__ import annotations

import contextlib
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

from .common import ROOT, sha256, system_info, utc_now, write_json
from .experiment_data import build_research_data, verify_research_data


class AccumulationWindow:
    def __init__(self, target):
        if target < 1:
            raise ValueError("Etkin batch pozitif olmali")
        self.target, self.samples = target, 0

    def add_batch(self, samples, *, last=False):
        if samples < 1 or samples > self.target:
            raise ValueError("Mikro-batch boyutu gecersiz")
        self.samples += samples
        if self.samples >= self.target or last:
            divisor, self.samples = self.samples, 0
            return divisor
        return None


def trim_uncommitted_csv(path, completed_epochs):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return
    kept = [rows[0]] + [row for row in rows[1:] if row and int(row[0].strip()) <= completed_epochs]
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(kept)


def protocol():
    return {"protocol_id": "hycado_stage1_development_v1", "model": "yolov9c.pt",
            "implementation": "Ultralytics 8.3.203 YOLOv9-C; no claim of original PGI training equivalence",
            "epochs": 80, "seed": 42, "imgsz": 800, "physical_batch": 2,
            "effective_batch_target": 8, "optimizer": "AdamW", "lr0": 0.0001,
            "lrf": 0.1, "cos_lr": True, "weight_decay": 0.0001, "warmup_epochs": 0.0,
            "amp": True, "workers": 0, "cache": False, "cpu_threads": 2,
            "mosaic": 0.25, "close_mosaic": 10, "translate": 0.1, "scale": 0.25,
            "fliplr": 0.5, "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.3,
            "split_seed": 20261022, "development_target_fraction": 0.10,
            "selection": "fixed epoch 80; EMA weights for inference; no best-by-proxy selection",
            "development_metrics": "Ultralytics diagnostic proxy; NOT official ignore-aware VisDrone AP",
            "official_val_policy": "not used by this training or its checkpoint choice",
            "accumulation": "explicit sample-weighted gradient mean; flush partial batch at epoch boundary",
            "resume": "FP32 train model + FP32 optimizer + EMA + scaler + scheduler; epoch boundary only",
            "epoch_randomness": "seed+epoch; reset loader generator and image buffer each epoch; not bitwise guaranteed",
            "old_experiment_artifacts_used": False}


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


LEGACY_CHECKPOINT_IO_CODE_SHA256 = "aa6fc330ff10530f5254dbca2d1aa51307d93804c96a2e92695258a46a8258aa"


def reconcile_protocol(run_dir, configuration):
    """Allow only the known first-checkpoint IO fix, before any committed state."""
    run_dir = Path(run_dir)
    path = run_dir / "protocol.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous == configuration:
        return
    old_hashes, new_hashes = previous.get("code_hashes", {}), configuration.get("code_hashes", {})
    old_other = {k: v for k, v in previous.items() if k != "code_hashes"}
    new_other = {k: v for k, v in configuration.items() if k != "code_hashes"}
    if (old_hashes.get("stage1.py") != LEGACY_CHECKPOINT_IO_CODE_SHA256 or
            set(old_hashes) != set(new_hashes) or old_other != new_other or
            any(old_hashes[k] != new_hashes[k] for k in old_hashes if k != "stage1.py") or
            (run_dir / "checkpoint_index.json").exists() or
            any((run_dir / "weights").glob("*.pt"))):
        raise ValueError("Sabit protokol/kod/veri degismis; mevcut deneyle karistirilmadi")
    backup = run_dir / "checkpoint_io_fix_backup"
    backup.mkdir(exist_ok=True)
    original = backup / "protocol.json"
    if original.exists() and json.loads(original.read_text(encoding="utf-8")) != previous:
        raise ValueError("Kaydetme duzeltmesi yedegi mevcut protokolle uyusmuyor")
    if not original.exists():
        shutil.copy2(path, original)
    for relative in ("result.json", "progress.json", "training/results.csv"):
        source = run_dir / relative
        target = backup / relative.replace("/", "_")
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    write_json(run_dir / "checkpoint_io_fix_migration.json", {
        "created_utc": utc_now(), "reason": "Windows fsync requires a writable checkpoint descriptor",
        "previous_protocol_sha256": fingerprint(previous),
        "new_protocol_sha256": fingerprint(configuration),
        "previous_code_sha256": old_hashes["stage1.py"],
        "new_code_sha256": new_hashes["stage1.py"],
        "committed_checkpoint_before_fix": False,
        "restart_epoch": 1, "training_settings_and_data_unchanged": True})
    print("Kaydetme duzeltmesi kaydedildi; veri ayrimi korunuyor. Egitim 1. epoch'tan baslayacak.", flush=True)


def finish_progress(run_dir, result):
    run_dir = Path(run_dir)
    path = run_dir / "progress.json"
    progress = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    index_path = run_dir / "checkpoint_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    progress.update(status=result["status"], last_committed_epoch=index.get("latest", {}).get("epoch", 0))
    if "error" in result:
        progress["error"] = result["error"]
    write_json(path, progress)
    return progress["last_committed_epoch"]


@contextlib.contextmanager
def training_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Bu deney baska bir pencerede calisiyor") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def memory_snapshot():
    result = {"created_utc": utc_now()}
    try:
        if os.name == "nt":
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page",
                                                     "avail_page", "total_virtual", "avail_virtual", "extended")]
            value = Status()
            value.length = ctypes.sizeof(value)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
                raise OSError("GlobalMemoryStatusEx failed")
            result.update(available_ram_gib=value.avail_phys/2**30,
                          available_commit_gib=value.avail_page/2**30, ram_load_percent=value.load)
        else:
            fields = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                fields[key] = int(value.split()[0])
            result["available_ram_gib"] = fields.get("MemAvailable", 0) / 2**20
    except (OSError, ValueError, AttributeError) as exc:
        result["memory_probe_error"] = str(exc)
    return result


def commit_checkpoint(run_dir, epoch, writer, protocol_hash):
    """Commit bytes, then pointer; interrupted writes never replace a valid pointer."""
    run_dir = Path(run_dir)
    index_path = run_dir / "checkpoint_index.json"
    previous = json.loads(index_path.read_text()) if index_path.exists() else None
    if previous and epoch <= previous["latest"]["epoch"]:
        raise ValueError("Kaydedilmis epoch'un uzerine yazilamaz")
    directory = run_dir / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"epoch_{epoch:04d}.pt"
    fd, temporary = tempfile.mkstemp(prefix="checkpoint_", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        writer(Path(temporary))
        # Windows _commit (os.fsync) rejects a descriptor opened read-only.
        with Path(temporary).open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        item = {"epoch": epoch, "path": target.relative_to(run_dir).as_posix(), "sha256": sha256(target)}
        write_json(index_path, {"protocol_sha256": protocol_hash, "latest": item,
                               "previous": previous["latest"] if previous else None})
    finally:
        Path(temporary).unlink(missing_ok=True)
    keep = {epoch, previous["latest"]["epoch"] if previous else epoch}
    for path in directory.glob("epoch_*.pt"):
        number = int(path.stem.rsplit("_", 1)[1])
        if number not in keep and number % 10 != 0:
            path.unlink()
    return item


def resume_path(run_dir, protocol_hash):
    run_dir = Path(run_dir)
    path = run_dir / "checkpoint_index.json"
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    if index["protocol_sha256"] != protocol_hash:
        raise ValueError("Checkpoint protokol kimligi uyusmuyor")
    latest = index["latest"]
    candidate = (run_dir / latest["path"]).resolve()
    if not candidate.is_relative_to(run_dir.resolve()) or sha256(candidate) != latest["sha256"]:
        raise ValueError("Checkpoint yolu/hash'i degismis")
    return candidate


def make_bundle(run_dir):
    run_dir = Path(run_dir)
    bundle = run_dir / "stage1_rapor.zip"
    allowed = {"protocol.json", "data/manifest.json", "data/label_audit.json", "progress.json",
               "history.json", "checkpoint_index.json", "result.json", "resolved_args.json",
               "environment.json", "model_identity.json", "code_hashes.json", "training/results.csv",
               "checkpoint_io_fix_migration.json", "checkpoint_io_fix_backup/protocol.json",
               "checkpoint_io_fix_backup/result.json", "checkpoint_io_fix_backup/training_results.csv"}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(allowed):
            path = run_dir / name
            if path.is_file():
                z.write(path, name)
        # A bounded log excerpt keeps a long training report small.
        log = run_dir / "console.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-150000))
                z.writestr("console_tail.log", stream.read())
    return bundle


def _train(run_dir, configuration, configuration_hash, checkpoint, *, _cpu_validation=False):
    import numpy as np
    import torch
    import ultralytics
    import cv2
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    if torch.__version__.split("+")[0] != "2.6.0" or ultralytics.__version__ != "8.3.203":
        raise RuntimeError("Sabitlenen torch=2.6.0 / ultralytics=8.3.203 gerekli")
    if not _cpu_validation and not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi")
    torch.set_num_threads(configuration["cpu_threads"])
    cv2.setNumThreads(1)
    def synchronize():
        if not _cpu_validation:
            torch.cuda.synchronize()

    def gpu_stat(name):
        return 0.0 if _cpu_validation else getattr(torch.cuda, name)()/2**30
    write_json(run_dir / "environment.json", {**system_info(True), "memory": memory_snapshot(),
               "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                            if d.metadata.get("Name")},
               "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads()})
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
    if saved and saved["protocol_sha256"] != configuration_hash:
        raise ValueError("Checkpoint ic protokolu uyusmuyor")
    if saved and saved["epoch"] + 1 >= configuration["epochs"]:
        return {"status": "COMPLETED", "completed_epochs": configuration["epochs"],
                "message": "Sabit epoch butcesi zaten tamamlandi; yeniden egitilmedi"}
    weights = ROOT / "work/fresh_official_pretrained" / configuration["model"]
    provenance_path = weights.with_suffix(".provenance.json")
    if not weights.exists() or not provenance_path.exists():
        raise FileNotFoundError("Ilk pilotta indirilen resmi agirlik ve kaynak kaydi gerekli")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if sha256(weights) != provenance["sha256"]:
        raise ValueError("Resmi baslangic agirligi hash'i degismis")
    if saved is None:
        model = YOLO(str(weights))
        initial_model = model.model
    else:
        model = None
        initial_model = saved["model"].float()
    history = list(saved["history"]) if saved else []
    trim_uncommitted_csv(run_dir / "training/results.csv", saved["epoch"]+1 if saved else 0)
    steps = {"actual": saved["actual_optimizer_steps"] if saved else 0,
             "attempted": saved["attempted_optimizer_steps"] if saved else 0}
    timers = {}

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(v) for v in value)
        return copy.deepcopy(value)

    class ResearchTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, 1 if mode == "val" else batch_size, rank, mode)
            loader.dataset.max_buffer_length = min(loader.dataset.max_buffer_length, 8)
            return loader

        def get_validator(self):
            validator = super().get_validator()
            validator.callbacks = self.callbacks
            return validator

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            self._current_batch_samples = int(batch["img"].shape[0])
            return batch

        def optimizer_step(self):
            # Native nbs=batch invokes this every micro-batch. Average explicit
            # per-image gradients and always flush at an epoch boundary.
            divisor = self._window.add_batch(self._current_batch_samples,
                                            last=self._batch_number == len(self.train_loader))
            if divisor is None:
                return
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(divisor)
            steps["attempted"] += 1
            super().optimizer_step()

        def resume_training(self, ckpt):
            if saved is None:
                return
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scaler.load_state_dict(saved["scaler"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.ema.ema.load_state_dict(saved["ema"].float().state_dict())
            self.ema.updates = saved["updates"]
            self.best_fitness = saved["best_fitness"]
            self.start_epoch = saved["epoch"] + 1
            if self.start_epoch > self.epochs - self.args.close_mosaic:
                self._close_dataloader_mosaic()
            print(f"DEVAM: tamamlanan epoch={self.start_epoch}, hedef={self.epochs}", flush=True)

        def save_model(self):
            if self._window.samples:
                raise RuntimeError("Epoch sonunda kaydedilmemis gradient birikimi")
            synchronize()
            current = {"epoch": self.epoch+1, "actual_optimizer_steps": steps["actual"]-timers["steps"],
                       "attempted_optimizer_steps": steps["attempted"]-timers["attempts"],
                       "train_seconds": timers["train_end"]-timers["start"],
                       "epoch_seconds_before_checkpoint": time.perf_counter()-timers["start"],
                       "loss_items": self.tloss.detach().cpu().tolist(), "memory": memory_snapshot(),
                       "proxy_development_metrics": {k: float(v) for k, v in self.metrics.items()},
                       "peak_reserved_gib": gpu_stat("max_memory_reserved"),
                       "peak_allocated_gib": gpu_stat("max_memory_allocated"),
                       "learning_rates": [g["lr"] for g in self.optimizer.param_groups]}
            if current["actual_optimizer_steps"] < 1:
                raise FloatingPointError("Epoch boyunca gercek optimizer guncellemesi olmadi")
            tensors = list(self.model.parameters()) + [v for state in self.optimizer.state.values()
                                                       for v in state.values() if torch.is_tensor(v)]
            if not all(bool(torch.isfinite(value.detach()).all()) for value in tensors):
                raise FloatingPointError("non-finite model/optimizer state")
            if any(not math.isfinite(v) for v in current["proxy_development_metrics"].values()):
                raise FloatingPointError("non-finite development metrics")
            payload = {"model": copy.deepcopy(unwrap_model(self.model)).float().cpu(),
                       "ema": copy.deepcopy(unwrap_model(self.ema.ema)).float().cpu(),
                       "epoch": self.epoch, "optimizer": cpu_copy(self.optimizer.state_dict()),
                       "scaler": self.scaler.state_dict(), "scheduler": self.scheduler.state_dict(),
                       "updates": self.ema.updates, "best_fitness": self.best_fitness,
                       "train_args": dict(vars(self.args)), "version": ultralytics.__version__,
                       "protocol_sha256": configuration_hash,
                       "history": history + [current], "actual_optimizer_steps": steps["actual"],
                       "attempted_optimizer_steps": steps["attempted"],
                       "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
            write_json(run_dir / "progress.json", {"status": "SAVING_CHECKPOINT", "epoch": self.epoch+1,
                       "training_and_development_finished": True,
                       "message": "Epoch is complete only after checkpoint commit succeeds"})
            item = commit_checkpoint(run_dir, self.epoch+1, lambda path: torch.save(payload, path), configuration_hash)
            history.append(current)
            write_json(run_dir / "history.json", history)
            write_json(run_dir / "progress.json", {"status": "RUNNING", "last_completed_epoch": self.epoch+1,
                       "checkpoint": item, "latest_epoch": current})
            print(f"CHECKPOINT: epoch {self.epoch+1}/{self.epochs}; {run_dir / item['path']}", flush=True)
            del payload
            gc.collect()

        def final_eval(self):
            # Do not strip optimizer state or select best.pt via the proxy metric.
            return None

    options = dict(model=str(weights), data=str(run_dir / "data/dataset.yaml"), task="detect",
                   epochs=configuration["epochs"], batch=configuration["physical_batch"],
                   nbs=configuration["physical_batch"], imgsz=configuration["imgsz"],
                   optimizer="AdamW", lr0=configuration["lr0"], lrf=configuration["lrf"],
                   cos_lr=True, weight_decay=configuration["weight_decay"], warmup_epochs=0.0,
                   amp=not _cpu_validation, device="cpu" if _cpu_validation else 0,
                   workers=0, cache=False, val=True, save=True,
                   save_period=-1, plots=False, patience=0, seed=configuration["seed"],
                   deterministic=False, multi_scale=False, rect=False,
                   project=str(run_dir), name="training", exist_ok=True, verbose=False,
                   mosaic=configuration["mosaic"], close_mosaic=configuration["close_mosaic"],
                   translate=configuration["translate"], scale=configuration["scale"],
                   fliplr=configuration["fliplr"], hsv_h=configuration["hsv_h"],
                   hsv_s=configuration["hsv_s"], hsv_v=configuration["hsv_v"],
                   degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
                   mixup=0.0, cutmix=0.0, copy_paste=0.0, conf=0.001, iou=0.6, max_det=500)
    trainer = ResearchTrainer(overrides=options)
    trainer.model = trainer.get_model(cfg=initial_model.yaml, weights=initial_model, verbose=False)
    del initial_model, model
    write_json(run_dir / "model_identity.json", {"official_initial_weights_sha256": sha256(weights),
               "model_class": type(trainer.model).__name__, "model_yaml": trainer.model.yaml,
               "parameters": sum(p.numel() for p in trainer.model.parameters()),
               "implementation_scope": configuration["implementation"]})

    def on_start(value):
        value._window = AccumulationWindow(configuration["effective_batch_target"])
        value._batch_number = 0
        if (value.batch_size != configuration["physical_batch"] or value.test_loader.batch_size != 1 or
                (not _cpu_validation and not value.amp)):
            raise RuntimeError("Istenen batch/AMP korunmadi")
        def actual_step(optimizer, args, kwargs):
            steps["actual"] += 1
        value.optimizer.register_step_post_hook(actual_step)
        write_json(run_dir / "resolved_args.json", vars(value.args))

    def epoch_start(value):
        if value._window.samples:
            raise RuntimeError("Onceki epoch'tan gradient kaldi")
        value._batch_number = 0
        seed = configuration["seed"] + value.epoch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = value.train_loader.dataset
        dataset.buffer.clear()
        dataset.ims = [None] * dataset.ni
        dataset.im_hw0 = [None] * dataset.ni
        dataset.im_hw = [None] * dataset.ni
        if value.train_loader.generator is not None:
            value.train_loader.generator.manual_seed(seed)
        value.train_loader.reset()
        gc.collect()
        synchronize()
        timers.update(start=time.perf_counter(), steps=steps["actual"], attempts=steps["attempted"])

    def batch_start(value):
        value._batch_number += 1

    def batch_end(value):
        if not bool(torch.isfinite(value.loss.detach()).all()):
            raise FloatingPointError("non-finite training loss")
        if value._batch_number % 250 == 0:
            write_json(run_dir / "progress.json", {"status": "RUNNING", "epoch": value.epoch+1,
                       "batch": value._batch_number, "batches_in_epoch": len(value.train_loader),
                       "actual_optimizer_steps": steps["actual"], "memory": memory_snapshot()})

    def epoch_end(value):
        synchronize()
        timers["train_end"] = time.perf_counter()

    for event, callback in [("on_train_start", on_start), ("on_train_epoch_start", epoch_start),
                            ("on_train_batch_start", batch_start), ("on_train_batch_end", batch_end),
                            ("on_train_epoch_end", epoch_end)]:
        trainer.add_callback(event, callback)
    trainer.train()
    return {"status": "COMPLETED", "completed_epochs": len(history),
            "cpu_validation_only": _cpu_validation,
            "actual_optimizer_steps": steps["actual"], "attempted_optimizer_steps": steps["attempted"],
            "amp_skipped_steps": steps["attempted"]-steps["actual"],
            "official_visdrone_metrics_computed": False,
            "reference": "fixed-budget final EMA checkpoint; not a proxy-selected best model"}


def run(prepared, run_dir=None):
    from .__main__ import _Tee
    run_dir = Path(run_dir or ROOT / "experiments/stage1_v1_seed42").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with training_lock(run_dir / ".training.lock"):
        with (run_dir / "console.log").open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(_Tee(__import__('sys').stdout, log)), contextlib.redirect_stderr(_Tee(__import__('sys').stderr, log)):
                print(f"\nGERCEK STAGE-1 EGITIMI: {utc_now()}\nDeney klasoru: {run_dir}", flush=True)
                try:
                    configuration = protocol()
                    if not (run_dir / "data").exists():
                        build_research_data(prepared, run_dir / "data")
                    verify_research_data(run_dir / "data")
                    configuration["dataset_manifest_sha256"] = sha256(run_dir / "data/manifest.json")
                    configuration["dataset_yaml_sha256"] = sha256(run_dir / "data/dataset.yaml")
                    expected_versions = {"torch": "2.6.0", "torchvision": "0.21.0", "ultralytics": "8.3.203",
                                         "numpy": "1.26.4", "opencv-python": "4.11.0.86", "Pillow": "11.3.0", "PyYAML": "6.0.2"}
                    versions = {name: importlib.metadata.version(name) for name in expected_versions}
                    if any(versions[name].split("+")[0] != version for name, version in expected_versions.items()):
                        raise RuntimeError("Sabitlenen temel paket surumleri degismis: " + str(versions))
                    configuration["runtime_versions"] = versions
                    files = ["stage1.py", "experiment_data.py", "data.py", "stability.py", "common.py"]
                    hashes = {name: sha256(Path(__file__).with_name(name)) for name in files}
                    configuration["code_hashes"] = hashes
                    digest = fingerprint(configuration)
                    protocol_path = run_dir / "protocol.json"
                    reconcile_protocol(run_dir, configuration)
                    write_json(protocol_path, configuration)
                    write_json(run_dir / "code_hashes.json", hashes)
                    checkpoint = resume_path(run_dir, digest)
                    if checkpoint:
                        print(f"Son tamamlanan epoch'tan devam: {checkpoint}", flush=True)
                    result = _train(run_dir, configuration, digest, checkpoint)
                except (Exception, KeyboardInterrupt) as exc:
                    traceback.print_exc()
                    from .pilot import classify_failure
                    result = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else classify_failure(str(exc)),
                              "error": str(exc), "error_type": type(exc).__name__, "memory": memory_snapshot(),
                              "resume": ("Rerun from the latest committed epoch" if (run_dir / "checkpoint_index.json").exists()
                                         else "No committed checkpoint; rerun restarts epoch 1 with the existing data split")}
                write_json(run_dir / "result.json", result)
                last_epoch = finish_progress(run_dir, result)
                print(f"SONUC: {result['status']}; kaydedilmis epoch: {last_epoch}/80", flush=True)
        bundle = make_bundle(run_dir)
        print(f"GONDERILECEK RAPOR: {bundle}", flush=True)
        return bundle, result["status"] == "COMPLETED"
