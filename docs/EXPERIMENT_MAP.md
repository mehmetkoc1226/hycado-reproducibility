# Historical execution map

All relative commands below are interpreted from `historical/experiments/`, where the package's `common.ROOT` resolves. These commands document the archived workflow. They are **not instructions to rerun the completed experiments now**.

| Stage | Implementation / invocation | Inputs and role |
|---|---|---|
| 02 | `python -m hycado_start prepare` | Official raw images/annotations; prepares derived labels |
| 05 | `python -m hycado_start stage1` | Fixed TRAIN/development grouping, pretrained YOLOv9-C; 80 rounds |
| 06-08 | `proposals.py`, `crop_merge.py`, `crop_data.py` | YOLO triggers, geometric merge, crop pixels/labels; see original numbered launchers |
| 09 | `crop_training_v2.py` | Calibration and AMP checks; pilot weights do not initialize final training |
| 10 | `python -m hycado_start.crop_experiment` | Crop catalog, original 09 calibration records, pinned recipe; 80 rounds, final EMA |
| 11-12 | `dev_evaluation.py`, `frozen_val.py` | Development diagnostics and frozen official VAL predictions |
| 17-18 | `full_image_pilot.py`, `python -m hycado_start.full_image_training` | Full-image RT-DETR; historical prerequisite identities, fresh COCO initialization, fixed 80 rounds |
| 19 | `python -m hycado_start.full_image_evaluation` | Final three checkpoints, fixed VAL records and comparators |
| 20 | `python -m hycado_start.joint_runtime` | Common batch-one timing session, three phases per method |
| 21 | `python -m hycado_start.fixed_grid_comparison` | Fixed-grid comparison and separate two-pass timing session |
| 22 | `python -m hycado_start.position_control` | Three relocated placements, fixed ordered crop counts/dimensions |
| 23 | `python -m hycado_start.paired_bootstrap` | Retained predictions and matching caches; paired image/proxy-cluster bootstrap |
| 24 | `metric_reference.py`, pinned upstream evaluator | Export and reference-source comparison; original verification ran under GNU Octave |
| 25 | `historical/provenance_audit.py` | Original images, labels, metadata and checkpoints; data/model identity audit |
| 26 | `analysis/stage26/full_image_fusion.py` from repository root | Five exact archives; CPU global-fusion prediction replay |

## Recorded GPU environment

The observed system used Python 3.11.5, torch 2.6.0+cu124, torchvision 0.21.0+cu124, Ultralytics 8.3.203, NumPy 1.26.4, Pillow 11.3.0 and OpenCV 4.11.0.86 on Windows build 26200 and an RTX 3060 Ti (8 GiB). The common timing record reports driver 610.60. Earlier setup reports used a different driver; they should not overwrite the timing-session identity.

The matching PyTorch wheel command is documented in the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/):

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/training.txt
```

Run that command from the repository root in an isolated training environment, then change into `historical/experiments/` to inspect the historical workflow. Direct versions are recorded; a complete transitive historical package lock has not been recovered. The separately pinned CPU plotting environment is not a substitute for this training environment.

## Exact repetition limits

The final states use seed 42 and fixed 80-round budgets. Historical recipes preserve the crop sampler, ignore-query classification mask, optimizer, learning-rate schedule, gradient accumulation, AMP and EMA. CUDA `grid_sample` backward was permitted with deterministic warnings; bitwise repeatability is not guaranteed.

Historical code validates exact upstream metadata and code hashes. In particular, `crop_experiment.validate_calibration` checks the retained calibration protocol/history and `configuration` checks catalog and dependency identities. Full-image training and evaluation depend on additional stage records. A fresh calibration run can generate different metadata even when scientific settings agree. Do not bypass these checks and describe the result as an already validated reproduction.

The included split text files specify exact membership. They do not replace sampler schedules, pixel identities or historical source-order fingerprints.
