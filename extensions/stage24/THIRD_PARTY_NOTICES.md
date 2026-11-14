# Third-party and retained components

The package includes the unchanged pycocotools 2.0.11 Windows CPython3.11 x64 wheel previously used in stage22/23. SHA256: 8e159232adae3aef6b4e2d37b008bff107b26e9ed3b48e70ea6482302834bd34. The original COCO license notice is retained in COCO_LICENSE.txt.

Sources: https://pypi.org/project/pycocotools/2.0.11/ and https://github.com/ppwwyyxx/cocoapi . The original API is https://github.com/cocodataset/cocoapi . COCOeval is used unchanged for new per-image matching and point evaluation. The retained stage23 accumulation helper is reference-tested against that API for bootstrap calculation.

SAHI is not bundled or reimplemented. The author's already installed P2 sahi_vendor, version 0.11.35, is reused. Original project: https://github.com/obss/sahi . Existing Ultralytics/PyTorch installations are reused; no dependency installation or replacement is performed.

The retained HyCADO modules/functions and their source identities are recorded in source_provenance.json. Toolkit AP is not relabeled as COCO AP. Checkpoints, original images, annotation contents and full prediction databases are not redistributed in this bundle.
