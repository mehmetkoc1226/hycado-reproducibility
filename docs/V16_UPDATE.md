# v16 completed-control update

This update preserves historical files and adds exact completed stages28–30 with immutable input/reference records, license notices and originally distributed package hashes. Only README and RELEASE_STATUS.md replace previous root documentation. All other old files retain their prior Git blob identities.

## Results and interpretation

- Two-resolution YOLO fusion:25.311 COCO AP100; HyCADO minus fusion −0.876 [−1.506,−0.152]. This is a later exploratory fixed control, not an equal-compute comparison.
- The entire five-point alpha grid is retained. Alpha0.125 reaches24.510; original0.25 reaches24.435; alpha1 reaches22.256. The original configuration remains0.25; no new CI or optimal-setting validation was computed for the grid.
- All1610 raw-query archives reproduce HyCADO-800, HyCADO-1280 and the800-region/1280-global control exactly.
- Under the specified COCO cap/ignore rules,4534 crop-filling display detections are FP at both IoU0.5 and0.75. FP includes duplicates and class/localization errors; no box-removal counterfactual or causal artefact effect was measured.

`paper_assets/v16` contains numerical evidence, old and new table checks, current text sources and numeric-only plots. The manuscript image-containing figures and atlas are not redistributed here. Stage25 metadata and export code remain available, but image/annotation/prediction bytes must come from their documented sources.

## Verification

The README lists the current publication-identity verifier and four self-contained recorded-result checks. `tools/verify_v16.py` preserves the original verifier while checking current documentation and all historical Git blob identities; the two documentation files are checked against their new identities, not skipped. The v16 control script recomputes twelve interval/SE pairs from retained draws, verifies source/code identities and all1610 output-index bindings, checks alpha values and geometry partitions, and reproduces six derived tables. It does not perform training, inference or new detection matching. Historical checks are retained independently.

Optional synthetic tests (in the isolated environment, with experiment_tests_v16.txt installed):

```bash
python -m unittest discover -s extensions/stage28 -p test_fusion28.py -v
python -m unittest discover -s extensions/stage29 -p test_audit29.py -v
python -m unittest discover -s extensions/stage30 -p test_alpha30.py -v
```

Original run wrappers target the recorded Windows64-bit Python3.11 project. The repository publication was checked in a separate Linux CPU environment; it is not a new Windows GPU or end-to-end training replication.

## Artifact availability

No checkpoints or full frozen prediction databases are added by this source update. Their recorded hashes and publication requirements remain in ARTIFACTS.md and the source manifests. No release URL or DOI is invented. Use the immutable commit permalink of this update for the completed v16 code and recorded results; the older21b07128402c621a37d9b5921143f21d105ed43e snapshot continues to identify the prior22–25 publication.

The extension manifest lists new/updated file SHA256 and Git blob identities, excluding its own recursive identity. It distinguishes source-only result verification from external prediction replay.
