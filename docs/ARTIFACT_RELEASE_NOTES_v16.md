# HyCADO checkpoints and frozen predictions accompanying v16

Code and recorded numerical results: https://github.com/mehmetkoc1226/hycado-reproducibility/tree/311bfe15c1a9df09326ec1d210b60c847905baed

The artifact packaging plan retains its original source snapshot, 21b07128402c621a37d9b5921143f21d105ed43e. This identifies packaging provenance; the v16 code snapshot above adds the completed stages 28–30.

## Contents

- Three exact retained final-round checkpoints: YOLOv9-C, crop-trained RT-DETR-L, and full-image RT-DETR-L. Original serialization is preserved.
- Four frozen prediction databases: stage 18 test-dev/relocation, stage 19 internal-development resolution extension, stage 20 test-dev resolution extension including retained raw crop queries, and stage 24 SAHI-640 test-dev.
- Five original historical replay inputs collected in legacy_replay_inputs.zip.
- release_manifest.json and SHA256SUMS.txt identify and hash the packaged assets. Each ZIP includes an artifact_manifest.json with member byte counts and SHA256 values.

The eight ZIPs total 2,876,308,641 bytes. The independently recorded expected ZIP hashes are in manifests/prepared_artifact_assets.json in the repository. That record describes local packaging verification, not evidence that public uploads have already completed.

## Scope

These files support prediction replay and access to retained model states. They do not include all historical experiment databases or establish a clean-clone end-to-end training replication. Public numerical-source checks were separately run from a fresh clone of commit 311bfe15c1a9df09326ec1d210b60c847905baed: identity verification and seven root tests passed.

Original VisDrone dataset images are not included. Obtain required data from the maintainers under their terms. Checkpoints may contain executable pickle serialization; use trusted files and the recorded compatible environment. Existing third-party dependency and pretrained-model terms remain applicable; this release introduces no blanket model or software license.
