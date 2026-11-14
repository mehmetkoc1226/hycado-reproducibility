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

The PyPI package is the maintained packaging/bugfix fork of the original COCO API. Wheel metadata identifies its license as FreeBSD. COCO_LICENSE.txt retains the upstream license notice. No authorship of COCOeval is claimed. The original COCO API is used for reference tests. uncertainty_core.py reproduces its 101-point accumulation on previously computed per-image matches and is tested against literal resampling with upstream COCOeval.

resampling.py retains the existing HyCADO filename/hash grouping, PCG64 frequency draw, and stable-tie replication helper functions from bootstrap_statistics.py. The historical toolkit AP statistic is not used as COCO AP. coco_core.py is unchanged from the completed stage22 package.

Recall-FP statistics reuse frozen toolkit-aligned global-max500 matches from the author's stage20 SQLite file. They are explicitly distinguished from COCO per-category maxDets100 average recall. No new annotation matching, detector inference, or score tuning is performed.

Official VisDrone images, labels, model checkpoints and full prediction databases are not redistributed in this bundle.
