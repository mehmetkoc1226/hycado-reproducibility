"""Extract the exact PyPI wheel into this package; never installs into the project environment."""
from pathlib import Path, PurePosixPath
import hashlib, platform, struct, sys, zipfile
HERE=Path(__file__).resolve().parent
NAME='pycocotools-2.0.11-cp311-cp311-win_amd64.whl'
SHA='8e159232adae3aef6b4e2d37b008bff107b26e9ed3b48e70ea6482302834bd34'
def ensure():
    if sys.platform!='win32' or sys.version_info[:2]!=(3,11) or struct.calcsize('P')!=8:
        raise RuntimeError('This launcher requires the recorded Windows x64 Python 3.11 environment.')
    import numpy
    wheel=HERE/'wheels'/NAME
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=SHA:
        raise RuntimeError('Bundled PyPI wheel identity differs.')
    target=HERE/'_vendor';target.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('pycocotools','pycocotools-2.0.11.dist-info'):
                raise RuntimeError('Unexpected wheel member.')
            p=target.joinpath(*rel.parts);data=z.read(name)
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.read_bytes()!=data:raise RuntimeError('Isolated evaluator files changed: '+name)
            else:p.write_bytes(data)
    sys.path.insert(0,str(target))
    import pycocotools.cocoeval as ev
    if not Path(ev.__file__).resolve().is_relative_to(target.resolve()):
        raise RuntimeError('Isolated evaluator not used.')
    return {'pycocotools':'2.0.11','wheel_sha256':SHA,'numpy':numpy.__version__,
            'cocoeval_sha256':hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest()}
