# Stage24 validation

13 tests pass on Linux/Python3.12/NumPy2.3.5 with the retained pycocotools2.0.11 evaluator. The tests cover:

- Exact original SAHI factory keyword arguments and the 10 expected class names.
- Original RGB decoding, 640-square slice geometry, overlap, standard prediction, NMS/IOU settings and CUDA synchronization calls through a mocked GPU adapter.
- Preservation of 1018-row SAHI output; canonicalization, image identity, payload hashes and corrupted cache rejection.
- Interruption after a committed image, then resume that performs inference only for the missing image; inference does not open annotation rows.
- Cache binding mismatch and explicitly closed SQLite connections, including Windows file-lifetime concerns.
- Original optimized toolkit matcher against the retained independent reference matcher.
- Original toolkit aggregation and global cap rules against the retained reference evaluator.
- COCO 101-point accumulation and literal image-replication tie handling against upstream COCOeval.
- Full coverage of the retained 1000 paired YOLO/HyCADO draws, reproducing the verified previous AP-difference interval.
- New SAHI bootstrap export, pairing and restart reuse of all 1000 completed synthetic draw records.

The SAHI model factory and inference function bodies were extracted without edits from the frozen P2 development script whose complete source SHA256 is 53ec58a853e3c93ca4363df9ec25df3fa08d808f5c5da31d1579a8cc979619b1. Matching and toolkit aggregation bodies likewise retain the previous implementation. source_provenance.json records the sources and hashes.

Actual model weights, Windows CUDA execution and the full real dataset are available only on the author's PC. GPU inference has NOT been executed here. The package reruns all tests, validates the actual checkpoint/environment, and requires five exact canonical numeric VAL replays before new test-dev inference. No tolerances or metric-based fallback settings are introduced.

No new SAHI accuracy, confidence interval, training result or speed benchmark is claimed by this preparation. Existing input files are read only; all new predictions and analyses are stored in a separately bound stage24 database.
