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
