# External artifacts

## Official data

Obtain VisDrone DET TRAIN (6471 source images) and official VAL (548) from https://github.com/VisDrone/VisDrone-Dataset. Role membership and recorded file/RGB/annotation SHA256 identities are included in `splits/`. The `images/train/...` names in these lists refer to the prepared project's logical source paths; they are not claims about the downloaded archive's directory layout.

Full images and raw annotations are not embedded in this source repository. The dataset's own distribution terms apply.

## Checkpoints

`manifests/checkpoints.json` records the exact selected checkpoint bytes and initialization identities. It supplies **no weight-download URL**. A hash verifies a file already obtained; it does not make that file available.

| Selected state | SHA256 |
|---|---|
| YOLOv9-C final EMA | `3b61a12a89a448b849bc965e44f00afa92042a5fc0928b4c4f801e0182f84149` |
| Crop RT-DETR-L final EMA | `fa6e3991b26650e2426a4bbac9253351d47087e2533d3b98d86a2f24b586cbf9` |
| Full-image RT-DETR-L final EMA | `0d7f046673037e83accccf8c430f6978799f59557a93f3be4a7ac2ea182eb57c` |

## Frozen-analysis inputs

Stage 26 requires exactly the five names and SHA256 values in `manifests/global_fusion_inputs.json`. The 24 input export (`24_metrik_dogrulama_rapor.zip`) and 24 reference-validation result (`HyCADO_24_Referans_Dogrulama_v1.zip`) are different files and both are required. Prediction archives are not replaced by the manuscript's metrics JSON.

Those retained input files exist in the authors' workflow, but public download locations have not yet been finalized. Consequently, the prepared repository supports self-contained table regeneration and source inspection; public end-to-end prediction replay remains dependent on releasing the external inputs. Full training additionally needs the historical stage records and original dataset/weights.

## Future public release record

When external files are published, add their actual URLs, byte sizes, existing SHA256 values and immutable release identifiers. Do not invent a DOI or use a changing branch URL as an immutable snapshot. Keep publication status explicit until the files are accessible.

## Test-dev and resolution-extension artifacts

The 18–20 protocols, metric summaries, intervals, source identities and prediction indexes are included under paper_assets/v12/evidence/extensions/. Exact inference sources are under extensions/stage18, stage19 and stage20. The report indexes identify retained outputs; they do not contain complete per-detection predictions or raw crop queries.

The large predictions.sqlite3 files remain external:

| Local experiment folder | Stage |
|---|---|
| experiments/testdev_frozen_v1_seed42 | 18 |
| experiments/full1280_development_v1_seed42 | 19 |
| experiments/testdev1280_extension_v1_seed42 | 20 |

Their exact recorded SHA256 values and pending-download status are in manifests/extension_prediction_inputs.json. Stage20 depends on the exact stage18/stage19 reports and databases, as well as the official data, selected checkpoints and original project prerequisites. Its source retains these checks; publishing a report does not make a fresh-clone GPU replay self-contained.

All official data remain with the dataset maintainers. Test-dev filenames and JPEG hashes are recorded in the stage18 dataset manifest. This records exact-file identities, not independently verified scene identities.

## Diagnostic source packages

The two small VAL diagnostic result reports are included and suffice for table regeneration and count consistency checks. The original stage16/stage17 diagnostic source ZIPs are retained externally for now; their identities are recorded in manifests/diagnostic_source_inputs.json. This update does not claim public distribution of those two source packages.
