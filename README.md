# HyCADO reproducibility materials

Companion materials for **HyCADO for Detector-Guided Region Refinement in UAV Object Detection**, Gul Erten and Mehmet Koc.

The current recorded-result revision is **v16**. It includes completed stages22–25 and28–30: COCO-style evaluation, paired uncertainty, SAHI-640 test-dev, frozen qualitative selection, two-resolution YOLO fusion, exact raw-crop replay, and descriptive score-multiplier/geometry diagnostics. This repository update runs no training or detector inference. Historical implementations and v12/v14 records remain intact.

## Current COCO results

VisDrone DET **test-dev,1610 images**, COCO-style bbox metrics with the recorded ignore-region preprocessing, **maxDets100 per image/category**. These scores are distinct from the historical toolkit scores further below.

| Method | AP | AP_small | AP_medium | AP_large | AR100 |
|---|---:|---:|---:|---:|---:|
| YOLO-1280 | 23.996 | 15.365 | 33.295 | 35.016 | 36.166 |
| HyCADO-1280, original alpha0.25 | 24.435 | 15.906 | 33.832 | 34.910 | 39.128 |
| SAHI-640 + same YOLO checkpoint | 24.529 | 15.692 | 32.793 | 41.419 | 39.817 |
| YOLO-800 + YOLO-1280 fusion | 25.311 | 15.661 | 35.160 | 43.238 | 39.520 |

HyCADO minus YOLO-1280: **+0.439 AP [0.161,0.838]**. HyCADO minus SAHI: **−0.095 [−0.776,0.743]**, which establishes neither superiority nor equivalence. HyCADO minus two-resolution fusion: **−0.876 [−1.506,−0.152]**. The latter control combines the same YOLO weights at targets800 and1280, leaves both score multipliers at1, and applies the cascade's final NMS/cap. No latency or equal-compute conclusion follows from these accuracy results.

Intervals are paired percentile intervals from1000 draws over61 filename/hash proxy groups, conditional on one trained seed42 model set. The split had already been examined. Size/class differences have no separate intervals. Low-FP recall is unchanged for the two1280 refinement variants at the reported budgets; SAHI and two-resolution fusion were not evaluated in that particular recall diagnostic.

## Descriptive multiplier and crop diagnostics

With weights, crop regions, decoder, score cutoff and NMS fixed:

| Crop-score multiplier alpha | COCO AP100 |
|---|---:|
| 0 — crop branch disabled; YOLO-1280 | 23.996 |
| 0.125 | 24.510 |
| **0.25 — original HyCADO** | **24.435** |
| 0.5 | 24.189 |
| 1 | 22.256 |

All five points are reported; **the original alpha0.25 is retained**. The0.075-point difference at0.125 has no newly computed CI and does not establish a newly validated optimal setting. This is descriptive post-hoc sensitivity, not an untouched confirmation set.

Stage29 verifies all1610 raw-query archives and exactly replays three frozen configurations. At alpha0.25,4534 evaluated crop-source display boxes (score>=0.05) occupy at least95% of their originating crop area; all are COCO FPs at IoU0.5 and0.75 after ignore filtering and the per-class top100 cap. A COCO FP may be a duplicate or a class/localization error. These counts are not an AP decomposition or a tested benefit from deleting those boxes. Stage29's raw geometry totals use a different, uncapped/unfiltered population.

## Reproduce recorded-result checks

Use a separate environment, not an active experiment environment:

```bash
python -m venv .venv-report
# Activate it for your platform, then:
python -m pip install -r requirements/report_v16.txt
python tools/verify_v16.py
python paper_assets/v16/scripts/check_extensions.py
python paper_assets/v16/scripts/check_val_diagnostics.py
python paper_assets/v16/scripts/build_current_results.py
python paper_assets/v16/scripts/build_revision_controls.py
```

These commands verify retained results, source/code identities, table bytes, class means and paired intervals. The v16 verifier checks the two updated documentation files against their new hashes and all other historical files against the prior published Git blob identities. The original `verify_release.py` remains an archival verifier of the initial documentation snapshot; use `verify_v16.py` for this checkout. **They do not load models or recompute AP from predictions.** The last command additionally verifies completed28–30 reports, twelve paired metric intervals/standard errors, alpha CSV values and geometry partitions, and six new table files. For numeric figure regeneration:

```bash
python -m pip install -r requirements/figures_v16.txt
python paper_assets/v16/scripts/build_revision_controls.py --write --figures
```

The included numeric figures show the multiplier grid and the retained full-range recall data on a common linear-near-zero/logarithmic-above0.5 axis. Dataset images, manuscript PDFs, the pipeline's image-containing artwork and the qualitative atlas remain outside this repository. The LaTeX files document the reporting source; they are not advertised as a self-contained image-complete Overleaf project.

## Original Windows CPU controls

Exact originally distributed packages, including references and package identities, are under:

- `extensions/stage28`: fixed YOLO-800/1280 fusion and paired comparisons;
- `extensions/stage29`: raw-query integrity, exact alpha0.25 replay and geometric provenance;
- `extensions/stage30`: fixed alpha grid and COCO-matched crop geometry.

They require the original frozen Windows Python3.11 project inputs. Their launchers use CPU only, do not install into the main environment, and resume signature-bound completed work. Inspect their `ONCE_OKUYUN_TR.md` and `protocol.json` first. **A fresh clone does not supply the external checkpoints, full prediction databases or original data.** Their download links remain pending. See [v16 update](docs/V16_UPDATE.md), [release status](docs/RELEASE_STATUS.md), and [external inputs](docs/ARTIFACTS.md).

## Historical v12 workflow

## Start here

Python 3.11 or newer is sufficient for the following CPU-only checks; no additional packages or GPU are needed:

```bash
python tools/verify_v16.py
python tools/verify_v12.py
python tools/reproduce_tables.py --output build/v12_tables
python -m unittest discover -s tests -v
```

The table command regenerates **11 LaTeX files** (tables and scalar definitions) from retained metrics and checks their exact SHA256 values against the v12 manuscript. It requires a new or empty output directory. This operation checks reporting consistency; it does not recompute AP from predictions.

The historical primary metric is the pinned VisDrone toolkit aggregation, written **AP_toolkit**. It is not COCO mAP. Class-macro AP is a separate diagnostic.

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

All conditions retain one seed-42 trained model set. Placement seeds are not training seeds. Bootstrap intervals describe sample-composition uncertainty for the fixed models using filename/hash proxy clusters. The 1280 test-dev extension was designed after the original test-dev results were inspected and is reported as **exploratory**. Secondary intervals are not multiplicity-adjusted. The historical v12 snapshot evaluated SAHI on VAL only; the completed stage24 test-dev result is reported above under the current extension.

## Repository map

- paper_assets/v12/: historical toolkit table sources, completed reports and verification/generation scripts.
- extensions/stage18/: original test-dev evaluation and matched relocation.
- extensions/stage19/: full 1280 cascade on 647 internal-development images.
- extensions/stage20/: fixed test-dev comparison of the two 1280 variants.
- extensions/p2_official_val/: retained higher-resolution/SAHI official-VAL package.
- extensions/stage14_cpu_fusion/: retained 800-region/1280-global development replay.
- historical/experiments/: original training, cropping, inference and control implementations.
- analysis/stage26/: original prediction-level global-fusion replay.
- splits/: exact role lists; test-dev JPEG identities are in the stage-18 dataset manifest.
- manifests/: source, checkpoint, archive and table identities.

The original files directly under paper_assets/ remain the historical snapshot. Use v12 for the historical toolkit table set, paper_assets/v14 for its frozen COCO extension, and paper_assets/v16 for the current recorded-result revision. Detailed timing tables and plots are excluded from the v12 table workflow; historical records remain available separately. The manuscript retains a brief hardware-bound cost disclosure and crop-call counts.

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
