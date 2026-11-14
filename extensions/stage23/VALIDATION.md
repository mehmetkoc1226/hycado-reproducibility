# Stage23 validation

14 tests pass on Linux/Python3.12/NumPy2.3.5 with pycocotools2.0.11. They check:

- Original point accumulation against official COCOeval, including absent categories and empty detections.
- Literal image replication under nonuniform frequencies, including exact tied TP/FP order within repeated images.
- Random multiclass fixtures with eight different replication vectors; precision-based metrics match at absolute tolerance 1e-10.
- COCO per-category cap100 and detections beyond a global rank cutoff; ignored match flags.
- Exact reproduction of the retained 61 filename/hash proxy groups and PCG64 draw scheme.
- Whole-score-tie recall selection and weighted recall equivalence to literal image replication, with variable resampled image count.
- Read-only COCO SQLite loading, entry-coverage changes, persistent draw reuse and digest corruption rejection.
- Stage20 frozen NPZ/SQLite recall adapter, TP/FP parity and source-byte preservation.
- Interval and CSV export coverage.

The Windows wheel is identical to the stage22 wheel already exercised by the author. Its Windows native extension cannot be executed here. All 14 tests are rerun on the target PC before full-data processing; full 1610-image point parity and frozen recall-count parity are mandatory before bootstrapping.

No real confidence interval or new research result has been computed here. Actual source databases remain on the author's computer. No detector is trained or invoked by this package.

Bootstrap metric scope is intentionally limited to COCO maxDets100/area=all and the three prespecified test-dev contrasts. Stage22 cap500 and size-specific point results are retained separately. Recall-FP uses the previously frozen toolkit-aligned global-max500 matches; it is not mislabeled as COCO AR100.
