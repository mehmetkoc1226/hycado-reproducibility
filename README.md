# HyCADO reproducibility materials

Companion materials for **HyCADO for Detector-Guided Region Refinement in UAV Object Detection**, Gul Erten and Mehmet Koc.

This repository contains the historical implementation and the recorded-result materials for manuscript revision v12. The update adds completed higher-resolution/sliced baselines, VAL diagnostics, test-dev evaluation, and the fixed 1280 extension. It does not run new training or inference.

## Start here

Python 3.11 or newer is sufficient for the following CPU-only checks; no additional packages or GPU are needed:

```bash
python tools/verify_release.py
python tools/verify_v12.py
python tools/reproduce_tables.py --output build/v12_tables
python -m unittest discover -s tests -v
```

The table command regenerates **11 LaTeX files** (tables and scalar definitions) from retained metrics and checks their exact SHA256 values against the v12 manuscript. It requires a new or empty output directory. This operation checks reporting consistency; it does not recompute AP from predictions.

The primary metric is the pinned VisDrone toolkit aggregation, written **AP_toolkit**. It is not COCO mAP. Class-macro AP is a separate diagnostic.

## What can be reproduced here?

| Task | Included materials | External inputs |
|---|---|---|
| Regenerate v12 numerical tables | Metric records, source-bound reports, generators, expected hashes | None |
| Verify completed 18–20 reports | Protocols, metrics, paired intervals, source/prediction identities | None for consistency checks |
| Inspect completed inference implementations | Exact extracted source packages under extensions/ | None for inspection |
| Replay historical global fusion from predictions | Stage-26 analysis and tests | Five exact files listed in manifests/global_fusion_inputs.json |
| Repeat historical training or detector inference | Historical implementation and fixed recipes | Official data, checkpoints, original prerequisite records and prediction databases |

Trained checkpoints and complete prediction databases are **not yet publicly hosted**. The original stage16/stage17 diagnostic source ZIPs also remain external; their small result reports and the table-generation/consistency code are included. Their recorded hashes identify the required bytes but are not download links. See [artifact requirements](docs/ARTIFACTS.md) and [release status](docs/RELEASE_STATUS.md).

## Accuracy results

Official VAL contains 548 images. All rows below use the same toolkit score.

| Condition | AP_toolkit |
|---|---:|
| YOLO-800 | 27.839 |
| YOLO-1280 | 31.914 |
| YOLO-1536 | 30.996 |
| YOLO + SAHI-640 | 32.254 |
| YOLO + SAHI-800 | 31.409 |
| HyCADO-800 | 29.987 |
| Global fusion, RT-DETR score multiplier 0.25 | 29.316 |
| Matched relocation mean | 28.448 |

On the 1610-image test-dev split:

| Condition | AP_toolkit |
|---|---:|
| YOLO-800 | 25.670 |
| HyCADO-800 | 26.768 |
| Matched relocation mean | 25.626 |
| YOLO-1280 | 28.738 |
| HyCADO-1280, 1280-based regions | 29.243 |
| 800-based regions + YOLO-1280 global detections | 29.281 |

The test-dev placement margin is **+1.142 AP**, paired proxy-cluster 95% interval **[+0.792, +1.452]**. The primary 1280 extension adds **+0.505 AP** over YOLO-1280, interval **[+0.129, +0.885]**. Retaining the 800-based regions adds **+0.544 AP**, interval **[+0.168, +0.924]**.

All conditions retain one seed-42 trained model set. Placement seeds are not training seeds. Bootstrap intervals describe sample-composition uncertainty for the fixed models using filename/hash proxy clusters. The 1280 test-dev extension was designed after the original test-dev results were inspected and is reported as **exploratory**. Secondary intervals are not multiplicity-adjusted. SAHI was evaluated on VAL; a SAHI test-dev result is not claimed.

## Repository map

- paper_assets/v12/: current table sources, completed reports and verification/generation scripts.
- extensions/stage18/: original test-dev evaluation and matched relocation.
- extensions/stage19/: full 1280 cascade on 647 internal-development images.
- extensions/stage20/: fixed test-dev comparison of the two 1280 variants.
- extensions/p2_official_val/: retained higher-resolution/SAHI official-VAL package.
- extensions/stage14_cpu_fusion/: retained 800-region/1280-global development replay.
- historical/experiments/: original training, cropping, inference and control implementations.
- analysis/stage26/: original prediction-level global-fusion replay.
- splits/: exact role lists; test-dev JPEG identities are in the stage-18 dataset manifest.
- manifests/: source, checkpoint, archive and table identities.

The original files directly under paper_assets/ remain the historical snapshot. Use the versioned v12 directory for current tables. Detailed timing tables and plots are excluded from the v12 table workflow; historical records remain available separately. The manuscript retains a brief hardware-bound cost disclosure and crop-call counts.

## Historical table and prediction replay

For the earlier four-file table/plot replay, create an isolated environment and install requirements/analysis.txt. Then:

```bash
python tools/reproduce_tables.py --revision legacy --output build/legacy_tables
```

The stage-26 unit tests require the analysis environment above:

```bash
python -m unittest discover -s analysis/stage26 -p test_fusion.py -v
```

For prediction-level global-fusion replay, obtain the five exact external files first:

```bash
python tools/verify_inputs.py --inputs INPUT_DIRECTORY
python analysis/stage26/full_image_fusion.py --inputs INPUT_DIRECTORY --output NEW_OUTPUT_DIRECTORY --workers 4
```

This CPU command applies the frozen fusion and toolkit-style scoring. It does not load model weights or repeat the historical Octave validation.

Read [the experiment map](docs/EXPERIMENT_MAP.md) and [v12 update notes](docs/V12_UPDATE.md) before using GPU launchers. They depend on historical project state and are not a validated fresh-clone end-to-end workflow. Keep this repository separate from an active experiment directory. Official VisDrone data must be obtained from the [dataset maintainers](https://github.com/VisDrone/VisDrone-Dataset). Third-party terms are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
