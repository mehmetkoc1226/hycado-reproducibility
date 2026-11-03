"""Isolated, real-data CUDA training capacity checks, not final experiments."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from .common import ROOT, sha256, system_info, utc_now, write_json
from .data import make_capacity_subset

ROLES = {
    "yolo800": {"model": "yolov9c.pt", "imgsz": 800},
    "rtdetr800": {"model": "rtdetr-l.pt", "imgsz": 800},
    "rtdetr960": {"model": "rtdetr-l.pt", "imgsz": 960},
}


def classify_failure(message):
    text = message.lower()
    if "opencv" in text and ("insufficient memory" in text or "outofmemoryerror" in text):
        return "HOST_OUT_OF_MEMORY"
    if "out of memory" in text or "cuda error: memory allocation" in text:
        return "OUT_OF_MEMORY"
    if any(token in text for token in ("nan", "non-finite", "not finite", "infeasible", "invalid numeric")):
        return "NUMERICAL_FAILURE"
    return "ERROR"


def child(job_file):
    """One child process owns one model/size/batch combination."""
    job_file = Path(job_file).resolve()
    job = json.loads(job_file.read_text(encoding="utf-8"))
    extended = job.get("kind") == "stability"
    result = {"purpose": "capacity_only_not_a_scientific_result", "created_utc": utc_now(),
              "role": job["role"], "requested_batch": job["batch"], "imgsz": job["imgsz"],
              "precision_requested": job["precision"], "status": "STARTED",
              "dataset_yaml_sha256": sha256(job["data"]),
              "subset_manifest_sha256": sha256(Path(job["data"]).with_name("manifest.json")),
              "rtdetr960_is_full_image_proxy": job["role"] == "rtdetr960"}
    if extended:
        result.update(purpose="stability_only_not_scientific_training", requested_epochs=job["epochs"],
                      requested_effective_batch=job["effective_batch"], epoch_records=[],
                      actual_optimizer_steps=0, attempted_optimizer_steps=0,
                      augmentation_policy="provisional_stability_v1", input_shapes=[])
    output = job_file.parent / "result.json"
    write_json(output, result)
    started = time.perf_counter()
    torch = None
    batch_start = None
    batch_times = []
    val_batches = 0
    model = None
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO, RTDETR
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.rtdetr.train import RTDETRTrainer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA bulunamadi; CPU'ya sessiz gecis yapilmadi")
        if torch.__version__.split("+")[0] != "2.6.0" or ultralytics.__version__ != "8.3.203":
            raise RuntimeError("Pilot icin torch=2.6.0 ve ultralytics=8.3.203 kullanin")
        result["environment"] = system_info(include_torch=True)
        result["total_gpu_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        result["free_gpu_gib_before_model"] = torch.cuda.mem_get_info(0)[0] / 2**30
        torch.cuda.reset_peak_memory_stats(0)

        class CapacityMixin:
            # Ultralytics normally doubles validation batch; keep validation at 1.
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                return super().get_dataloader(dataset_path, batch_size=1 if mode == "val" else batch_size,
                                              rank=rank, mode=mode)

            def get_validator(self):
                validator = super().get_validator()
                validator.callbacks = self.callbacks
                return validator

            def save_model(self):
                # Capacity weights must never be mistaken for scientific checkpoints.
                return None

            def final_eval(self):
                # Epoch validation already ran; no checkpoint to evaluate again.
                return None

            def preprocess_batch(self, batch):
                batch = super().preprocess_batch(batch)
                if extended:
                    shape = list(batch["img"].shape)
                    if shape not in result["input_shapes"]:
                        result["input_shapes"].append(shape)
                    result["max_targets_in_train_batch"] = max(
                        result.get("max_targets_in_train_batch", 0), int(len(batch["cls"])))
                return batch

            def optimizer_step(self):
                if extended:
                    result["attempted_optimizer_steps"] += 1
                return super().optimizer_step()

        class YoloCapacityTrainer(CapacityMixin, DetectionTrainer):
            pass

        class RtdetrCapacityTrainer(CapacityMixin, RTDETRTrainer):
            pass

        pretrained_dir = ROOT / "work" / "fresh_official_pretrained"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        weights = pretrained_dir / job["model"]
        provenance_file = weights.with_suffix(".provenance.json")
        if weights.exists():
            if not provenance_file.exists():
                raise RuntimeError("Kaydi olmayan mevcut agirlik reddedildi: " + str(weights))
            provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
            if provenance["sha256"] != sha256(weights):
                raise RuntimeError("On egitimli agirligin hash degeri degismis")
        constructor = YOLO if job["role"] == "yolo800" else RTDETR
        # The exact known official asset filename is resolved by pinned Ultralytics.
        model = constructor(str(weights))
        if not provenance_file.exists():
            write_json(provenance_file, {"downloaded_utc": utc_now(), "sha256": sha256(weights),
                       "filename": weights.name, "source": "official Ultralytics pretrained asset resolver",
                       "resolver_version": ultralytics.__version__})
        result["initial_weights_sha256"] = sha256(weights)

        def on_train_start(trainer):
            result["actual_batch"] = int(trainer.batch_size)
            result["amp_effective"] = bool(trainer.amp)
            result["validation_batch"] = int(trainer.test_loader.batch_size)
            result["train_examples"] = len(trainer.train_loader.dataset)
            result["validation_examples"] = len(trainer.test_loader.dataset)
            result["resolved_args"] = json.loads(json.dumps(vars(trainer.args), default=str))
            if extended:
                # AMP skips optimizer.step on overflow; count real updates, not requests.
                def step_completed(optimizer, args, kwargs):
                    result["actual_optimizer_steps"] += 1
                trainer.optimizer.register_step_post_hook(step_completed)
                result["train_loader_batches_per_epoch"] = len(trainer.train_loader)
                result["validation_loader_batches_per_epoch"] = len(trainer.test_loader)
                if trainer.batch_size != job["batch"] or trainer.test_loader.batch_size != 1:
                    raise RuntimeError("Istenen train/validation batch korunmadi")
                if not trainer.amp:
                    raise RuntimeError("Kararlilik testi AMP acik gerektirir")
                write_json(output, result)

        epoch_started = None
        epoch_train_ended = None
        previous_steps = 0

        def on_epoch_start(trainer):
            nonlocal epoch_started
            if extended:
                torch.cuda.synchronize()
                epoch_started = time.perf_counter()

        def on_epoch_train_end(trainer):
            nonlocal epoch_train_ended
            if extended:
                torch.cuda.synchronize()
                epoch_train_ended = time.perf_counter()

        def on_fit_epoch_end(trainer):
            nonlocal previous_steps
            if not extended:
                return
            # Check the learned state as well as finite reported losses/metrics.
            state_tensors = list(trainer.model.parameters())
            state_tensors += [v for state in trainer.optimizer.state.values()
                              for v in state.values() if torch.is_tensor(v)]
            if not all(bool(torch.isfinite(v.detach()).all()) for v in state_tensors):
                raise FloatingPointError("non-finite model or optimizer state")
            updates = result["actual_optimizer_steps"] - previous_steps
            previous_steps = result["actual_optimizer_steps"]
            if updates < 1:
                raise FloatingPointError("No actual optimizer update in completed epoch")
            torch.cuda.synchronize()
            record = {"epoch": int(trainer.epoch) + 1, "actual_optimizer_steps": updates,
                      "train_seconds_including_data": epoch_train_ended - epoch_started,
                      "epoch_seconds_including_validation": time.perf_counter() - epoch_started,
                      "peak_reserved_gib_so_far": torch.cuda.max_memory_reserved(0) / 2**30,
                      "loss_items": trainer.tloss.detach().cpu().tolist()}
            result["epoch_records"].append(record)
            print("KARARLILIK EPOCH:", json.dumps(record), flush=True)
            write_json(output, result)

        def on_batch_start(trainer):
            nonlocal batch_start
            torch.cuda.synchronize()
            batch_start = time.perf_counter()

        def on_batch_end(trainer):
            if trainer.loss is not None and not bool(torch.isfinite(trainer.loss.detach()).all()):
                raise FloatingPointError("non-finite training loss")
            torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - batch_start)
            result["completed_train_batches"] = len(batch_times)
            if extended:
                if trainer.accumulate * trainer.batch_size != job["effective_batch"]:
                    raise RuntimeError("Istenen gradient accumulation korunmadi")
                if len(batch_times) % 25 == 0:
                    write_json(output, result)

        def on_val_batch_end(validator):
            nonlocal val_batches
            val_batches += 1

        def on_val_end(validator):
            values = getattr(validator.metrics, "results_dict", {})
            if any(not __import__("math").isfinite(float(v)) for v in values.values()):
                raise FloatingPointError("non-finite validation metrics")

        def on_train_end(trainer):
            result["optimizer_state_entries"] = len(trainer.optimizer.state)
            result["accumulate"] = int(trainer.accumulate)
            result["actual_batch"] = int(trainer.batch_size)
            result["completed_val_batches"] = val_batches

        for event, callback in (("on_train_start", on_train_start),
                                ("on_train_batch_start", on_batch_start),
                                ("on_train_batch_end", on_batch_end),
                                ("on_val_batch_end", on_val_batch_end),
                                ("on_val_end", on_val_end), ("on_train_end", on_train_end),
                                ("on_train_epoch_start", on_epoch_start),
                                ("on_train_epoch_end", on_epoch_train_end),
                                ("on_fit_epoch_end", on_fit_epoch_end)):
            model.add_callback(event, callback)
        train_options = dict(
            model=str(weights), task="detect", mode="train", pretrained=True,
            data=job["data"], imgsz=job["imgsz"], batch=job["batch"], epochs=1,
            nbs=job["batch"], optimizer="AdamW", lr0=0.0001, weight_decay=0.0001,
            warmup_epochs=0.0, amp=job["precision"] == "amp", device=0,
            workers=0, cache=False, val=True, save=False, plots=False,
            project=str(job_file.parent), name="capacity_run", exist_ok=False,
            seed=42, deterministic=False, patience=0, verbose=False,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
            degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
            perspective=0.0, flipud=0.0, fliplr=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            max_det=500, conf=0.001, iou=0.6,
        )
        if extended:
            from .stability import stability_options
            train_options.update(stability_options(job))
        # Direct trainer avoids Model.train() reloading a deliberately absent
        # best.pt after our capacity-only run. No pilot checkpoint is produced.
        trainer_class = YoloCapacityTrainer if job["role"] == "yolo800" else RtdetrCapacityTrainer
        trainer = trainer_class(overrides=train_options, _callbacks=model.callbacks)
        trainer.model = trainer.get_model(cfg=model.model.yaml, weights=model.model, verbose=False)
        trainer.train()
        if not result.get("optimizer_state_entries") or not result.get("completed_val_batches"):
            raise RuntimeError("Optimizer veya validation kontrolu tamamlanmadi")
        if extended:
            if (len(result["epoch_records"]) != job["epochs"] or
                    len(batch_times) != job["epochs"] * result["train_loader_batches_per_epoch"] or
                    val_batches != job["epochs"] * result["validation_loader_batches_per_epoch"]):
                raise RuntimeError("Istenen tum epoch/train/validation adimlari tamamlanmadi")
            result["amp_skipped_optimizer_steps"] = (result["attempted_optimizer_steps"] -
                                                     result["actual_optimizer_steps"])
        if result["actual_batch"] != job["batch"]:
            result["status"] = "BATCH_CHANGED"
        elif job["precision"] == "amp" and not result.get("amp_effective"):
            result["status"] = "PRECISION_CHANGED"
        else:
            result["status"] = "PASS_STABILITY_ONLY" if extended else "PASS_CAPACITY_ONLY"
    except BaseException as exc:
        result["status"] = classify_failure(str(exc))
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        traceback.print_exc()
    finally:
        result["wall_seconds_including_setup"] = time.perf_counter() - started
        if batch_times:
            steady = batch_times[2:] or batch_times
            result["median_train_batch_seconds_after_warmup"] = statistics.median(steady)
        if torch is not None and torch.cuda.is_available():
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(0) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(0) / 2**30
            total = result.get("total_gpu_gib")
            if total:
                result["reserved_fraction_of_total"] = result["peak_reserved_gib"] / total
        write_json(output, result)
    return 0 if result["status"] in {"PASS_CAPACITY_ONLY", "PASS_STABILITY_ONLY"} else 1


def run(prepared, report_dir, batches=(1, 2), roles=None, precision="amp", timeout=1800):
    report_dir = Path(report_dir).resolve()
    run_dir = report_dir / ("pilot_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True)
    dataset = make_capacity_subset(prepared, run_dir / "capacity_data")
    roles = list(ROLES) if roles is None else roles
    results = []
    for role in roles:
        for batch in batches:
            if batch < 1:
                raise ValueError("Batch pozitif olmali")
            job_dir = run_dir / f"{role}_batch{batch}_{precision}"
            job_dir.mkdir()
            job = {**ROLES[role], "role": role, "batch": batch, "precision": precision,
                   "data": str(dataset)}
            job_file = job_dir / "job.json"
            write_json(job_file, job)
            print(f"\nTEST: {role}, batch={batch}, {precision}. Gunluk: {job_dir / 'console.log'}", flush=True)
            try:
                with (job_dir / "console.log").open("w", encoding="utf-8") as stream:
                    completed = subprocess.run([sys.executable, "-m", "hycado_start", "_probe_child",
                                                "--job", str(job_file)], cwd=ROOT,
                                               stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                path = job_dir / "result.json"
                result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "ERROR"}
                if result.get("status") == "STARTED":
                    result["status"] = "PROCESS_FAILED"
                result["exit_code"] = completed.returncode
                if completed.returncode and result.get("status") == "PASS_CAPACITY_ONLY":
                    result["status"] = "PROCESS_FAILED"
            except subprocess.TimeoutExpired:
                result = {"status": "TIMEOUT", "timeout_seconds": timeout}
            result.update(role=role, requested_batch=batch, precision_requested=precision,
                          job_directory=str(job_dir))
            results.append(result)
            print("SONUC:", result["status"], flush=True)
            write_json(run_dir / "summary.json", {"purpose": "capacity_only", "results": results})
    lines = ["HyCADO yerel kapasite testi", "", "Bunlar makale deneyleri veya performans sonuclari degildir.",
             "RT-DETR 960 testi tam goruntu vekilidir; gercek crop ve augmentation ile yeniden sinanmalidir.",
             "Bellek sayilari PyTorch allocator'una aittir; baska GPU uygulamalarinin bellegini kapsamaz.", ""]
    for result in results:
        lines.append(f"{result['role']} batch={result['requested_batch']} {precision}: {result['status']}; "
                     f"peak_reserved_GiB={result.get('peak_reserved_gib', 'NA')}")
    lines += ["", "PASS_CAPACITY_ONLY: bu kisa veri ve ayarlarda egitim + validation bitti.",
              "OUT_OF_MEMORY: hedef ayar sigmadi. ERROR/NUMERICAL_FAILURE bellek hatasi diye yorumlanmaz.",
              "Tam calisma karari icin en yogun gercek crop'lar ve nihai augmentation da test edilmelidir."]
    (run_dir / "OZET.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Compact return file, without the images or experimental weights.
    import zipfile
    bundle = report_dir / (run_dir.name + "_rapor.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name in {"summary.json", "OZET.txt", "job.json", "result.json", "console.log"}:
                archive.write(path, path.relative_to(run_dir))
    print(f"\nGonderilecek rapor: {bundle}", flush=True)
    return bundle
