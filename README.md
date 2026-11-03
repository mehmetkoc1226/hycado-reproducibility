# HyCADO reproducibility materials

Companion materials for **HyCADO for Detector-Guided Region Refinement in UAV Object Detection**, Gul Erten and Mehmet Koc.

Source repository: https://github.com/mehmetkoc1226/hycado-reproducibility

This repository publishes the source snapshot and recorded-result reproduction materials. Public hosting of the trained weights and frozen prediction inputs remains pending; no complete external-artifact release or DOI is asserted.

## Reproduction scope

| Task | Included here | Additional inputs | Validation status |
|---|---|---|---|
| Check file identities and split membership | Source and manifests | None | Checked during package preparation |
| Regenerate numerical paper tables and two plots | Frozen metric master and generator | CPU plotting environment | Four table files reproduced byte for byte |
| Recalculate the two global-fusion conditions | Unchanged stage-26 engine | Five hash-identified input files | Original execution retained; unit checks rerun for this package |
| Inspect training, cropping, inference, grid, relocation and bootstrap implementations | Historical code snapshot and recipes | Original data, checkpoints and stage records | Source identity checked; no fresh GPU run performed here |
| Re-run the full historical experiment chain | Historical implementation | Full original project state, official data, weights and prerequisite records | Not a validated fresh-clone workflow |

Rendering a table from recorded numbers is **not** a prediction replay or a training replication. The primary metric is literal pinned VisDrone toolkit AP (`AP_toolkit`), not COCO mAP. The separately reported class-macro diagnostic is also not COCO mAP.

## Quick start without a GPU

From this repository's root, using Python 3.12:

```bash
python tools/verify_release.py
python -m venv .venv-analysis
```

Activate the environment (`.venv-analysis\Scripts\activate` on Windows, `source .venv-analysis/bin/activate` on Linux/macOS), then:

```bash
python -m pip install -r requirements/analysis.txt
python tools/reproduce_tables.py --output build/table_replay
python -m unittest discover -s tests -v
python -m unittest discover -s analysis/stage26 -p test_fusion.py -v
```

The table command checks all four generated `.tex` files against the frozen SHA256 manifest and writes `validation.json` beside the generated tables/plots. It refuses a nonempty output directory. Use another output path for another replay.

## Prediction-level global-fusion replay

Place the five files listed in `manifests/global_fusion_inputs.json` in one directory. Download locations for these retained experiment archives have **not yet been published**; the manifest identifies the required bytes without claiming public availability.

```bash
python tools/verify_inputs.py --inputs INPUT_DIRECTORY
python analysis/stage26/full_image_fusion.py --inputs INPUT_DIRECTORY --output NEW_OUTPUT_DIRECTORY --workers 4
```

This CPU analysis reads frozen predictions and annotations, applies fixed RT score multipliers 1.0 and 0.25, then performs the recorded fusion and toolkit-style scoring. It does not load detector weights or measure latency. Historical reference parity under GNU Octave is a separate retained result; this command does not rerun it.

## Data and historical experiments

Official VisDrone DET data must be obtained from the [dataset maintainers](https://github.com/VisDrone/VisDrone-Dataset). This repository does not redistribute full images, labels, checkpoints or raw prediction archives. `splits/` contains exact filename lists and the recorded image/annotation identities for 5824 training, 647 development and 548 official-VAL images. These are filename/hash groups, not independently verified scenes.

`historical/experiments/` preserves the code, recipes and launchers from the evaluated project. Read [the execution map](docs/EXPERIMENT_MAP.md) before using them. The crop trainer checks the original calibration protocol, data catalog, upstream source hashes and environment. Later stages also check earlier records. Freshly generated timestamps or machine-dependent records may fail these guards. They have deliberately not been removed: removing a guard would create a new execution path that requires its own validation.

Do not copy this repository over a running experiment directory. The historical GPU launchers are not required to inspect or reproduce the paper tables.

## Results and limits

| Condition | AP_toolkit |
|---|---:|
| YOLOv9-C | 27.839 |
| Full-image RT-DETR-L | 19.760 |
| Global fusion, RT score multiplier 1.0 | 24.394 |
| Global fusion, RT score multiplier 0.25 | 29.316 |
| Fixed grid + YOLO | 28.810 |
| Relocation mean, three placements | 28.448 |
| HyCADO | 29.987 |

These results concern seed-42 models and 548 previously examined VAL images. Crop-placement seeds are not training seeds. Bootstrap intervals condition on the retained models. The measured cascade averages 1116.504 ms/image on the recorded RTX 3060 Ti system; no global-fusion FPS is inferred by adding component times.

See [the release status](docs/RELEASE_STATUS.md), [external artifact requirements](docs/ARTIFACTS.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
