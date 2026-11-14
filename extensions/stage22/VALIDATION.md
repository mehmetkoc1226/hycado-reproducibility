# Validation record — v2 repair

## Confirmed source of the failure

The received 22B error stops in input_inventory.validate_rows at the universal 1000-row check. The pinned P2 VAL index has two SAHI-640 records above 1000; its maximum is 1018. The original P2 source caps YOLO outputs at 1000 but has no such cap for the merged SAHI archive. Both exceptional records are listed in repair_record.json. No detector failure or training failure is indicated by this traceback.

## Repair boundaries

- Only SAHI-640/800 use their exact frozen per-image index count instead of a global 1000-row cap. File SHA256, image identity, all row values, geometry, class and score checks remain active.
- YOLO and HyCADO source caps remain 1000. No prediction is truncated, reordered or re-scored.
- All 2740 P2 VAL files are now hash-checked and all their rows validated before metric scoring, including the dense images missed by the old 12-image metric preflight.
- Scientific protocol.json and coco_core.py are byte-identical to v1. Original data, predictions and checkpoint files are not changed.
- New implementation/cache namespace supplementary_coco_v2_seed42 preserves v1 outputs and avoids mixing signatures. The combined launcher runs a new preflight then the full calculation automatically.

## Tests completed here

All 13 tests pass with Linux/Python3.12/NumPy2.3.5/pycocotools2.0.11. The original nine tests cover upstream precision/recall parity, independent cap100/cap500 runs, score ties, empty outputs, native area boundaries, ignore preprocessing and cache/resume integrity. Four additional regression tests cover:

1. A valid 1018-row SAHI archive is accepted unchanged while count mismatch, nonfinite values, zero-area boxes and invalid classes are rejected.
2. YOLO/cascade 1000-row caps remain enforced; unknown methods are rejected.
3. The actual Sources.get VAL adapter loads a 1018-row SAHI file unchanged, leaves source bytes unchanged, and rejects file hash tampering.
4. A minority-class true positive beyond global rank 1000 survives to official COCO's per-category caps. Streaming output still equals direct upstream COCOeval precision/recall tensors at absolute tolerance 1e-12.

## Actual target confirmation

The author already ran the original nine tests successfully on Windows/Python3.11/NumPy1.26.4 and confirmed the pinned Windows wheel imports. This Linux workspace cannot execute that Windows extension. The new combined launcher reruns all 13 tests on the target before scoring. Real prediction payloads remain on the author's computer; complete real-data validation is performed there by the launcher.

This repair produces no new full-dataset scientific result here. Bootstrap confidence intervals are still pending after point scoring.

