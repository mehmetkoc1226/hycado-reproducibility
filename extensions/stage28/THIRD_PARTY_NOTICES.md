# Third-party components

This bundle includes the unmodified PyPI wheel:
pycocotools-2.0.11-cp311-cp311-win_amd64.whl
SHA256: 8e159232adae3aef6b4e2d37b008bff107b26e9ed3b48e70ea6482302834bd34
Bytes: 80519

Source and distribution information:
- https://pypi.org/project/pycocotools/2.0.11/
- https://pypi.org/pypi/pycocotools/2.0.11/json
- https://github.com/ppwwyyxx/cocoapi
- https://github.com/cocodataset/cocoapi

The PyPI package is the maintained packaging/bugfix fork of the original COCO API. Wheel metadata identifies its license as FreeBSD. COCO_LICENSE.txt retains the upstream license notice. No authorship of COCOeval is claimed. Its code is not replaced by a custom approximate AP formula.

visdrone_preprocess.py is an unchanged copy of the previously retained HyCADO dev_metrics.py. Only its annotation/ignore helpers are used in the new COCO path; historical AP functions are not changed or reused as COCO AP.

Official VisDrone images, labels, model checkpoints and full prediction databases are not redistributed in this bundle.
