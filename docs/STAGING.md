# External artifact staging (not a public weight release)

The five stage-26 inputs were collected and independently SHA256/ZIP checked for
the accompanying distribution bundle. Their public download URL remains pending.
The three selected `.pt` weights are not included in this source checkout or the
companion bundle. Source hashes are in `manifests/checkpoints.json` and
`manifests/global_fusion_inputs.json`.

On the authors' original Windows project, with the five input files in one
directory, use Python 3.11 (standard library only):

```text
py -3.11 tools/stage_external_artifacts.py --project D:\HyCADO_Temiz_Baslangic --inputs D:\HyCADO_Yayin_Hazirligi --output D:\HyCADO_Temiz_Baslangic\release_stage --stage
py -3.11 tools/stage_external_artifacts.py --verify-staged --output D:\HyCADO_Temiz_Baslangic\release_stage
```

Expected weight locations are `experiments/stage1_v1_seed42/weights/epoch_0080.pt`,
`experiments/rtdetr_crop_v1_seed42/weights/epoch_0080.pt`, and
`experiments/rtdetr_full_image_v1_seed42/weights/epoch_0080.pt`. The script
verifies exact hashes and copies eight files into a separate staging directory;
it does not upload, retrain, run inference, or replace original files. Its
local manifest has byte sizes and SHA256 values; the public URL field is null.

Publication will require real accessible external download links, license/rights
review for the derived weights and source images, and an independent download
and CPU replay check. A source repository commit by itself is not a complete
external-artifact release.
